"""Advance a single cross-group request with durable outbound actions.

The sender is injected: it receives (platform, bot_id, scene_id, text,
reply_to) and returns a confirmed platform message ID. This module never
retries a send whose result is unknown.
"""

import asyncio
from uuid import uuid4
from datetime import datetime, timezone
from collections.abc import Callable, Awaitable

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from nonebot_plugin_orm import get_session

from ..model import BrainTask, BrainWait, BrainAction, BrainCandidateDecision
from .actions import ActionSpec, confirm, prepare, claim_send
from .repository import claim, finish, accept_reply, finish_expired, wait_for_reply, fail_before_send, claim_finalization, pending_reply_candidates
from .coordinator import plan_request, classify_reply
from .observations import configured_space
from ..message_metadata import is_reliable_platform_id

Sender = Callable[[str, str, str, str, str | None], Awaitable[str]]
Clock = Callable[[], datetime]
Spaces = dict[str, list[str]] | Callable[[], dict[str, list[str]]]


def _spaces(value: Spaces) -> dict[str, list[str]]:
    return value() if callable(value) else value


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def _block(task_id: str, token: str, now: datetime) -> None:
    async with get_session() as db:
        await db.execute(update(BrainTask).where(
            BrainTask.task_id == task_id, BrainTask.status == "running", BrainTask.claim_token == token,
        ).values(status="blocked", claim_token=None, lease_until=None,
                 revision=BrainTask.revision + 1, updated_at=now))
        await db.commit()


async def _confirm_direct_send(task_id: str, token: str, *, platform: str, bot_id: str, source: str,
                               source_message: str, target: str, spaces: Spaces, sender: Sender, clock: Clock) -> str:
    """Journal the source acknowledgment after the target send is confirmed."""
    if not configured_space(_spaces(spaces), source):
        await _block(task_id, token, clock())
        return "blocked"
    async with get_session() as db:
        notice, _ = await prepare(db, ActionSpec(task_id, "send_notice", "notice", platform, source,
                                                 f"已向 {target} 群发送消息。"), task_token=token, now=clock())
        action_id, status, receipt = notice.action_id, notice.status, notice.receipt_message_id
        await db.commit()
    if status in {"sending", "delivery_unknown"}:
        await _block(task_id, token, clock())
        return "delivery_unknown"
    if status == "prepared":
        send_token = uuid4().hex
        async with get_session() as db:
            if not await claim_send(db, action_id, task_token=token, send_token=send_token, now=clock()):
                await db.rollback()
                return "unavailable"
            await db.commit()
        try:
            receipt = await sender(platform, bot_id, source, f"已向 {target} 群发送消息。", source_message)
        except Exception:
            await _block(task_id, token, clock())
            return "delivery_unknown"
        if not is_reliable_platform_id(receipt):
            await _block(task_id, token, clock())
            return "delivery_unknown"
        async with get_session() as db:
            if not await confirm(db, action_id, send_token=send_token, message_id=receipt, now=clock()):
                await db.rollback()
                return "delivery_unknown"
            await db.commit()
    elif status != "confirmed" or not is_reliable_platform_id(receipt):
        raise ValueError("Source notice cannot be delivered")
    async with get_session() as db:
        completed = await finish(db, task_id, token, now=clock())
        await db.commit()
    return "completed" if completed else "unavailable"


async def process_candidate(task_id: str, *, spaces: Spaces, model, clock: Clock = utcnow) -> str:
    """Classify at most one quoted candidate; persist decisions before retrying."""
    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        waiting = await db.get(BrainWait, task_id)
        if task is None or waiting is None or task.status != "waiting_input":
            return "unavailable"
        space = configured_space(_spaces(spaces), task.source_scene)
        if space is None or configured_space(_spaces(spaces), waiting.target_scene) != space:
            return "blocked"
        action = await db.scalar(select(BrainAction).where(
            BrainAction.task_id == task_id, BrainAction.logical_step == "ask", BrainAction.status == "confirmed",
            BrainAction.receipt_message_id == waiting.question_message,
        ))
        candidates = await pending_reply_candidates(db, task_id=task_id, now=clock())
        if action is None or not candidates:
            return "unavailable"
        candidate = candidates[0]
        candidate_id, question, text, unquoted = candidate.id, action.content, candidate.text, candidate.reply_to is None
        ambiguous = False
        if unquoted:
            overlapping = await db.scalar(select(func.count()).select_from(BrainWait).join(
                BrainTask, BrainTask.task_id == BrainWait.task_id,
            ).where(BrainTask.bot_id == task.bot_id, BrainTask.status == "waiting_input",
                    BrainWait.platform == candidate.platform, BrainWait.target_scene == candidate.scene_id,
                    BrainWait.status == "open", BrainWait.unquoted_from <= candidate.created_at,
                    BrainWait.unquoted_until >= candidate.created_at))
            ambiguous = (overlapping or 0) > 1

    try:
        decision = "uncertain" if ambiguous else await asyncio.wait_for(
            classify_reply(model, question=question, candidate=text, unquoted=unquoted), 60,
        )
    except Exception:
        # Remember an inconclusive attempt so one bad model response cannot
        # trigger a new paid call on every scheduler tick.
        decision = "uncertain"

    async with get_session() as db:
        task = await db.get(BrainTask, task_id)
        waiting = await db.get(BrainWait, task_id)
        if task is None or waiting is None or task.status != "waiting_input":
            return "unavailable"
        space = configured_space(_spaces(spaces), task.source_scene)
        if space is None or configured_space(_spaces(spaces), waiting.target_scene) != space:
            return "blocked"
        candidates = await pending_reply_candidates(db, task_id=task_id, now=clock())
        candidate = next((item for item in candidates if item.id == candidate_id), None)
        if candidate is None:
            return "unavailable"
        try:
            async with db.begin_nested():
                db.add(BrainCandidateDecision(task_id=task_id, observation_id=candidate_id, decision=decision, created_at=clock()))
                await db.flush()
        except IntegrityError:
            return "unavailable"
        if decision == "answer":
            accepted = await accept_reply(db, task_id, platform=candidate.platform, target_scene=candidate.scene_id,
                                          reply_to=candidate.reply_to or "", message_id=candidate.message_id,
                                          user_id=candidate.user_id, text=candidate.text, now=clock(), unquoted=unquoted)
            if not accepted:
                await db.rollback()
                return "unavailable"
        await db.commit()
    return decision


async def run_finalization(task_id: str, *, spaces: Spaces, sender: Sender, clock: Clock = utcnow) -> str:
    """Deliver an expired task's pre-recorded notice without resending unknowns."""
    token = uuid4().hex
    async with get_session() as db:
        if not await claim_finalization(db, task_id, token, now=clock()):
            return "unavailable"
        task = await db.get(BrainTask, task_id, populate_existing=True)
        action = await db.scalar(select(BrainAction).where(
            BrainAction.task_id == task_id, BrainAction.logical_step == "timeout", BrainAction.effect_slot == "notice",
        ))
        if task is None or action is None:
            await db.rollback()
            return "unavailable"
        platform, bot_id, source, source_message, failure_reason = task.platform, task.bot_id, task.source_scene, task.source_message, task.failure_reason
        action_id, target, text, status = action.action_id, action.target_scene, action.content, action.status
        await db.commit()
    if not configured_space(_spaces(spaces), source) or target != source:
        return "blocked"
    if status in {"sending", "delivery_unknown"}:
        return "delivery_unknown"
    if status == "prepared":
        send_token = uuid4().hex
        async with get_session() as db:
            if not await claim_send(db, action_id, task_token=token, send_token=send_token, now=clock()):
                await db.rollback()
                return "unavailable"
            await db.commit()
        try:
            receipt = await sender(platform, bot_id, source, text, source_message)
        except Exception:
            return "delivery_unknown"
        if not is_reliable_platform_id(receipt):
            return "delivery_unknown"
        async with get_session() as db:
            if not await confirm(db, action_id, send_token=send_token, message_id=receipt, now=clock()):
                await db.rollback()
                return "delivery_unknown"
            await db.commit()
    elif status != "confirmed":
        return "blocked"
    async with get_session() as db:
        completed = await finish_expired(db, task_id, token, now=clock())
        await db.commit()
        return ("failed" if failure_reason == "planning_failed" else "expired") if completed else "unavailable"


async def run_task(task_id: str, *, spaces: Spaces, model, sender: Sender,
                   clock: Clock = utcnow, target_names: dict[str, str] | None = None,
                   name_resolver: Callable[[str, str, tuple[str, ...]], Awaitable[dict[str, str]]] | None = None) -> str:
    """Advance a queued task to waiting, completed, blocked or unavailable."""
    token = uuid4().hex
    async with get_session() as db:
        if not await claim(db, task_id, token, now=clock()):
            return "unavailable"
        task = await db.get(BrainTask, task_id, populate_existing=True)
        waiting = await db.get(BrainWait, task_id)
        assert task is not None
        snapshot = (task.platform, task.bot_id, task.source_scene, task.source_message, task.request, task.deadline)
        answer = (waiting.answer_text, waiting.target_scene) if waiting and waiting.status == "answered" else None
        direct = await db.scalar(select(BrainAction).where(BrainAction.task_id == task_id, BrainAction.logical_step == "send"))
        step = "return" if answer else "send" if direct is not None else "ask"
        previous = direct if step == "send" else await db.scalar(select(BrainAction).where(BrainAction.task_id == task_id, BrainAction.logical_step == step))
        prior = (previous.action_id, previous.platform, previous.target_scene, previous.content,
                 previous.status, previous.receipt_message_id) if previous else None
        await db.commit()

    platform, bot_id, source, source_message, request, deadline = snapshot
    current_spaces = _spaces(spaces)
    space = configured_space(current_spaces, source)
    if space is None:
        await _block(task_id, token, clock())
        return "blocked"
    allowed = tuple(group for group in current_spaces[space] if configured_space(current_spaces, group) == space)
    try:
        if prior is not None:
            action_id, action_platform, target, text, status, receipt_id = prior
        else:
            if answer:
                answer_text, answer_scene = answer
                if not answer_text or answer_scene not in allowed:
                    raise ValueError("Answer source is no longer in this space")
                target, text = source, f"{answer_scene} 群回复：{answer_text}"
            else:
                try:
                    if name_resolver is not None:
                        try:
                            target_names = await asyncio.wait_for(name_resolver(platform, bot_id, allowed), 4)
                        except Exception:
                            pass
                    proposal = await asyncio.wait_for(plan_request(model, request=request, source_scene=source, target_scenes=allowed,
                                                                    target_names=target_names), 60)
                except Exception:
                    async with get_session() as db:
                        recorded = await fail_before_send(db, task_id, token, now=clock())
                        await db.commit()
                    return "finalizing" if recorded else "unavailable"
                target = proposal.target_scene if proposal.kind == "ask" else source
                text = proposal.text
            assert target is not None
            async with get_session() as db:
                action, _ = await prepare(db, ActionSpec(task_id, step, "message", platform, target, text),
                                          task_token=token, now=clock())
                action_id, action_platform, target, text, status, receipt_id = (
                    action.action_id, action.platform, action.target_scene, action.content, action.status, action.receipt_message_id,
                )
                await db.commit()
        current_spaces = _spaces(spaces)
        current_space = configured_space(current_spaces, source)
        if (action_platform != platform or current_space != space or configured_space(current_spaces, target) != space
                or target not in allowed or (answer and target != source) or (step == "send" and target == source)):
            raise ValueError("Action target no longer permitted")
        if status in {"sending", "delivery_unknown"}:
            await _block(task_id, token, clock())
            return "delivery_unknown"
        if status == "prepared":
            send_token = uuid4().hex
            async with get_session() as db:
                if not await claim_send(db, action_id, task_token=token, send_token=send_token, now=clock()):
                    await db.rollback()
                    return "unavailable"
                await db.commit()
            try:
                receipt_id = await sender(platform, bot_id, target, text, source_message if target == source else None)
            except Exception:
                # Leave sending in the journal. Lease recovery marks it unknown.
                await _block(task_id, token, clock())
                return "delivery_unknown"
            if not is_reliable_platform_id(receipt_id):
                await _block(task_id, token, clock())
                return "delivery_unknown"
            async with get_session() as db:
                if not await confirm(db, action_id, send_token=send_token, message_id=receipt_id, now=clock()):
                    await db.rollback()
                    return "delivery_unknown"
                await db.commit()
        elif status != "confirmed" or not is_reliable_platform_id(receipt_id):
            raise ValueError("Action cannot be delivered")

        if step == "send":
            return await _confirm_direct_send(task_id, token, platform=platform, bot_id=bot_id, source=source,
                                              source_message=source_message, target=target, spaces=spaces,
                                              sender=sender, clock=clock)
        async with get_session() as db:
            if target == source:
                completed = await finish(db, task_id, token, now=clock())
                await db.commit()
                return "completed" if completed else "unavailable"
            assert receipt_id is not None
            ready = await wait_for_reply(db, task_id, token, platform=platform, target_scene=target,
                                         question_message=receipt_id, deadline=deadline, now=clock())
            await db.commit()
            return "waiting_input" if ready else "unavailable"
    except Exception:
        await _block(task_id, token, clock())
        return "blocked"
