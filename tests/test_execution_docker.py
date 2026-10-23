"""Real Docker regression tests; use a locally built executor image, never pull."""

import io
import os
import base64
import shutil
import asyncio
import tarfile

import pytest


@pytest.fixture(scope="module")
async def docker_image():
    from nonebot_plugin_ai_groupmate.agent.execution_backend import _command

    if not shutil.which("docker"):
        pytest.skip("Docker CLI not installed")
    image = os.environ.get("AI_GROUPMATE_TEST_EXECUTION_IMAGE", "ai-groupmate-executor:local")
    try:
        status, output, _ = await asyncio.wait_for(_command("docker", "info", "--format", "{{.OSType}}"), timeout=5)
        if status or output.strip() != b"linux":
            pytest.skip("Linux Docker daemon not available")
        status, _, _ = await asyncio.wait_for(_command("docker", "image", "inspect", image), timeout=5)
        if status:
            pytest.skip(f"Build the executor image first: {image}")
    except (OSError, asyncio.TimeoutError):
        pytest.skip("Docker daemon not available")
    return image


@pytest.mark.parametrize("with_files", [False, True])
async def test_docker_tmpfs_round_trip(docker_image, with_files, monkeypatch):
    from nonebot_plugin_ai_groupmate.agent import execution_backend as module

    request: dict = {"language": "python", "code": "print('inline')"}
    if with_files:
        request.update(
            code="from pathlib import Path\nimport sys\nPath('out/nested').mkdir()\nPath('out/nested/result.bin').write_bytes(Path('inputs/nested/input.bin').read_bytes())\nprint('warning', file=sys.stderr)",
            files=[{"name": "nested/input.bin", "data": base64.b64encode(b"input\x00\xff").decode()}],
        )
        original_archive = module.input_archive

        def owned_archive(request):
            # Exercise extraction without CAP_CHOWN, even for foreign archive owners.
            output = io.BytesIO()
            with tarfile.open(fileobj=io.BytesIO(original_archive(request))) as source, tarfile.open(fileobj=output, mode="w") as target:
                for item in source:
                    item.uid = item.gid = 1000
                    target.addfile(item, source.extractfile(item) if item.isfile() else None)
            return output.getvalue()

        monkeypatch.setattr(module, "input_archive", owned_archive)
    result = await module.run_docker(request, image=docker_image)
    assert result["exit_code"] == 0, result
    assert result["stdout"] == ("" if with_files else "inline\n")
    assert result["stderr"] == ("warning\n" if with_files else "")
    assert module.decode_files(result["files"], limit=100) == ([("nested/result.bin", b"input\x00\xff")] if with_files else [])


async def test_docker_deleted_output_reports_tar_error(docker_image):
    from nonebot_plugin_ai_groupmate.agent.execution_backend import ExecutionError, run_docker

    with pytest.raises(ExecutionError, match="无法收集输出文件.*tar:"):
        await run_docker({"language": "python", "code": "import os\nos.rmdir('out')"}, image=docker_image)
