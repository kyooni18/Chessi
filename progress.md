Original prompt: huggingface 모델을 로드해서, 웹에서 가볍게 AI와 체스 게임을 즐기게 하고 싶어
시작할 때 HF에서 kyooni/chessi-0.1 모델 로드하고, AI와의 대국할 수 있게

- 2026-02-13: 초기 분석. 현재 저장소는 PyQt GUI 중심이며 웹 앱 코드가 없음.
- 2026-02-13: 구현 계획: Python 백엔드에서 HF 모델 사전 로드 + 정적 웹 UI(체스판/AI 응수) 추가.
- 2026-02-13: `web/server.py` 추가. Flask API로 게임 상태/모델 로딩/AI 수 생성 엔드포인트 구현.
- 2026-02-13: `web/static/index.html`, `web/static/styles.css`, `web/static/app.js` 추가. 웹 체스판 렌더링, 클릭 이동, 자동 AI 응수, 모델 상태 표시 구현.
- 2026-02-13: 기본 HF repo를 `kyooni/chessi-0.1`로 통일 (`Chessi.py`, `chess_lm_gui.py`, `test.py`).
- 2026-02-13: `requirements.txt`에 `Flask` 추가, README에 웹 서버 실행/환경변수 문서화.
- 2026-02-13: `Flask` 미설치 환경 문제로 `web/server.py`를 표준 라이브러리(`http.server`) 기반으로 교체. 추가 pip 의존성 없이 실행 가능하도록 수정.
- 2026-02-13: 실기동 중 `kyooni/chessi-0.1` 로드 실패 확인. 웹 서버에 자동 폴백(`kyooni18/chessi-0.1`) 로딩 로직 및 warning 상태 추가.
- 2026-02-13: PyQt/CLI 기본 repo는 기존(`kyooni18/chessi-0.1`)으로 복원해 기존 흐름 회귀 방지.
- 2026-02-13: 구문 검증 통과 (`python3 -m py_compile web/server.py`, `node --check web/static/app.js`).
- 2026-02-13: `CHESSI_WEB_PORT=8787` 실기동/API 검증 완료. `kyooni/chessi-0.1` 실패 후 `kyooni18/chessi-0.1` 폴백 로드 확인, `/api/player-move`에서 자동 AI 응수(`e4 -> c5`) 확인.
- 2026-02-13: `develop-web-game` Playwright Node 클라이언트 시도했으나 이 환경은 Node `playwright` 패키지 해석 불가(네트워크 차단으로 설치 불가).
- 2026-02-13: Python Playwright 대체 검증 완료. 사람 수/자동 AI 응수/흑 선택 시 AI 선착수/console error 없음 확인, 스크린샷 `output/web-game/ui-playwright.png` 갱신.
- 2026-02-13: UI 개선: 모델 재로드 버튼 줄바꿈 수정, Moves 번호 중복 표시 수정(`ol`->`ul`).

TODO / Next suggestions:
- 브라우저에서 `kyooni/chessi-0.1`이 실제 공개되면 폴백 없이도 READY가 되는지 재확인.
- 필요 시 `/api/reload-model`에 `force_download` 토글 UI 추가.
- 선택적으로 mobile 레이아웃에서 보드/상태 패널 비율 추가 미세조정.
- 2026-02-13: 사용자 요청에 따라 웹 아키텍처를 서버 API 의존에서 브라우저 온디바이스 실행으로 전환.
- 2026-02-13: `web/static/app.js`를 클라이언트 단독 체스 엔진(chess.js CDN) + ONNX 온디바이스 추론 루프로 교체. 서버 `/api/*` 호출 제거.
- 2026-02-13: 모델 로딩 인디케이터(loading/ready/error), 체크메이트/스테일메이트/드로우 종료 오버레이, 마지막 수 애니메이션 적용.
- 2026-02-13: `scripts/export_webgpu_onnx.py` 추가 (HF 모델 -> `web/model/chessi.onnx` + vocab export).
- 2026-02-13: README 웹 실행 절차를 "정적 파일 + 온디바이스 추론" 방식으로 업데이트.
- 2026-02-13: 웹 UI에 애니메이션 추가: 마지막 수 하이라이트 플래시, 착수 기물 드롭 애니메이션, 타깃 칸 펄스.
- 2026-02-13: 모델 로딩 인디케이터 추가(loading/ready/error). 모델 불가 시 로컬 fallback AI 메시지 표시.
- 2026-02-13: 종료 판정 오버레이 추가(체크메이트/스테일메이트/각종 드로우), 오버레이 클릭 시 새 게임 리셋.
- 2026-02-13: 서버 API 의존 제거. 프론트 단독 실행(chess.js + onnxruntime-web)으로 전환.
- 2026-02-13: Playwright 검증: 체크메이트/스테일메이트 FEN 주입 시 오버레이 정상 출력 확인 (`output/web-game/ondevice-status-test.png`).
- 2026-02-13: AI 선택 로직 업데이트: model top-k를 300으로 증가하고, 생성 후보 전부를 복제 보드에서 SAN 합법성 검사 후 첫 합법 수를 채택하도록 변경.
- 2026-02-13: `render_game_to_text`에 `ai_candidate_trace`(generated/checked/legalCount/chosen) 추가.
- 2026-02-13: 브라우저 검증에서 fallback AI 경로 기준 `ai_candidate_trace` 확인 완료.
