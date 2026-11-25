# Hermes에서 TensorPath v0.2.1 사용하기

압축 해제 후 프로젝트 디렉터리에서 다음을 실행합니다. 기존 Hermes와 LM Studio는 그대로 사용합니다. Python 3.10 이상이 필요합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[mcp,setup]'
tensorpath --root examples/sample_traces setup --project .
```

마지막 명령은 미리보기이며 파일을 수정하지 않습니다. Hermes를 종료하고 확인 후 적용합니다.

```bash
tensorpath --root examples/sample_traces setup --project . --apply
tensorpath --root examples/sample_traces doctor --protocol
```

기존 TensorPath 설정/스킬과 충돌한다면 `--replace`를 붙여 미리보기 후 `--replace --apply`로 적용합니다. 기존 설정과 교체된 스킬은 `~/.hermes/tensorpath-backup-*`에 보관됩니다. 다른 MCP 서버 설정은 유지됩니다. 적용 중에는 다른 프로세스에서 설정을 수정하지 마세요. 별도 Herdr Pane은 필요하지 않습니다.

Hermes를 다시 실행하고 자연어로 요청합니다.

> TensorPath로 healthy와 detached를 비교해서 gradient 단절 근거를 확인하고 HTML 보고서를 만들어줘.

보고서는 기본적으로 trace root의 `reports` 폴더에 생성됩니다. MCP가 보고서 경로를 반환하며 브라우저는 자동 실행하지 않습니다.

## 직접 만든 모델 분석

자신의 프로젝트와 저장 위치를 지정하여 등록합니다. 예시 경로는 실제 환경에 맞게 수정합니다.

```bash
tensorpath --root ~/tensorpath-traces setup --project ~/my-model --replace
tensorpath --root ~/tensorpath-traces setup --project ~/my-model --replace --apply
```

이후 Hermes에서 모델 탐색 및 최소 Forward 기록 수집을 요청합니다. 기존 Trace가 있다면 원래 모델이나 PyTorch가 없어도 됩니다. 새로운 Trace가 필요하면 모델을 실행하는 Python 환경에 TensorPath를 설치해야 합니다. 이미 PyTorch가 있다면 `python -m pip install -e /실제/패키지/경로`로 충분합니다. Agent는 모델 실행 전에 구체적인 코드와 명령, 입력, 부작용을 설명하고 승인을 요청하도록 설계했습니다. 이는 Skill 지침이며 강제 샌드박스가 아닙니다.

```python
from tensorpath import capture_forward
path = capture_forward(model, "/absolute/traces", args=(x,))
```

Forward-only는 backward나 optimizer.step을 호출하지 않습니다. 다만 사용자 모델 자체의 forward 부작용까지 방지하지는 못합니다. Gradient 분석은 실제 미분 가능한 loss를 구성해 Recorder로 기록합니다. 자세한 예제는 README 및 examples를 참조하세요.

## 검증과 제한

`doctor --protocol`은 실제 MCP stdio 연결과 도구 탐색을 검사합니다. Hermes에서의 실제 호출과 모델의 Skill 준수 여부는 별도로 확인해야 합니다. MCP 연결이 불가능하면 CLI로도 분석/시각화할 수 있습니다. PyTorch Eager Module 수준의 기록이며 실제 operator dataflow, 전체 backward graph, MLX 모델의 내부 기록은 미지원입니다.
