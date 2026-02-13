const boardEl = document.querySelector("#board");
const modelIndicatorEl = document.querySelector("#model-indicator");
const resultOverlayEl = document.querySelector("#result-overlay");
const resultTitleEl = document.querySelector("#result-title");
const resultSubtitleEl = document.querySelector("#result-subtitle");

const pieceToImage = {
  P: "/static/pieces/wp.svg",
  N: "/static/pieces/wn.svg",
  B: "/static/pieces/wb.svg",
  R: "/static/pieces/wr.svg",
  Q: "/static/pieces/wq.svg",
  K: "/static/pieces/wk.svg",
  p: "/static/pieces/bp.svg",
  n: "/static/pieces/bn.svg",
  b: "/static/pieces/bb.svg",
  r: "/static/pieces/br.svg",
  q: "/static/pieces/bq.svg",
  k: "/static/pieces/bk.svg",
};

const modelConfig = {
  repoId: "kyooni18/chessi-0.1",
  modelUrl: window.CHESSI_MODEL_URL || "/model/chessi.onnx",
  vocabUrl: window.CHESSI_VOCAB_URL || "/model/vocab.json",
  inputDType: window.CHESSI_MODEL_INPUT_DTYPE || "int32",
  fixedInputSeqLen: Number(window.CHESSI_MODEL_FIXED_SEQ_LEN || 32),
  maxSeqLen: 256,
  topK: 300,
};

const state = {
  chess: null,
  ChessClass: null,
  selectedSquare: null,
  legalTargets: [],
  busy: false,
  lastMoveUci: null,
  animateMoveUci: null,
  model: null,
  aiCandidateTrace: {
    generated: 0,
    checked: 0,
    legalCount: 0,
    chosen: null,
  },
  modelState: {
    phase: "loading",
    message: "Model loading...",
    detail: "",
  },
};

function squareName(row, col) {
  const files = "abcdefgh";
  return `${files[col]}${8 - row}`;
}

function cleanMoveToken(raw) {
  if (!raw) {
    return null;
  }

  let token = String(raw).trim();
  if (!token) {
    return null;
  }

  if (["1-0", "0-1", "1/2-1/2", "*", "..."].includes(token)) {
    return null;
  }

  if (/^\d+\.(\.\.)?$/.test(token)) {
    return null;
  }

  token = token.replace(/^\d+\.(\.\.)?/, "").trim();
  token = token.replace(/^\.+|\.+$/g, "");
  token = token.replace(/0-0-0/g, "O-O-O").replace(/0-0/g, "O-O");
  token = token.replace(/[!?]+$/g, "");

  if (!token || ["1-0", "0-1", "1/2-1/2", "*"].includes(token)) {
    return null;
  }

  return token;
}

function humanTurn(chess) {
  return chess.turn() === "w";
}

function canInteract(chess) {
  if (!chess || state.busy) {
    return false;
  }
  if (state.modelState.phase === "loading") {
    return false;
  }
  if (deriveGameStatus(chess).state !== "ongoing") {
    return false;
  }
  return humanTurn(chess);
}

function parseLastMoveSquares(lastMoveUci) {
  if (!lastMoveUci || lastMoveUci.length < 4) {
    return [];
  }
  return [lastMoveUci.slice(0, 2), lastMoveUci.slice(2, 4)];
}

function pieceSymbolAt(chess, square) {
  const piece = chess.get(square);
  if (!piece) {
    return "";
  }
  const type = piece.type.toLowerCase();
  return piece.color === "w" ? type.toUpperCase() : type;
}

function boardRows(chess) {
  const rows = [];
  for (let rank = 8; rank >= 1; rank -= 1) {
    const row = [];
    for (const file of "abcdefgh") {
      row.push(pieceSymbolAt(chess, `${file}${rank}`));
    }
    rows.push(row);
  }
  return rows;
}

function legalTargets(chess) {
  const mapping = {};
  for (const move of chess.moves({ verbose: true })) {
    if (!mapping[move.from]) {
      mapping[move.from] = [];
    }
    if (!mapping[move.from].includes(move.to)) {
      mapping[move.from].push(move.to);
    }
  }
  for (const fromSquare of Object.keys(mapping)) {
    mapping[fromSquare].sort();
  }
  return mapping;
}

function deriveGameStatus(chess) {
  const isCheckmate =
    (typeof chess.isCheckmate === "function" && chess.isCheckmate()) ||
    (typeof chess.in_checkmate === "function" && chess.in_checkmate());
  if (isCheckmate) {
    const winner = chess.turn() === "w" ? "black" : "white";
    return {
      state: "checkmate",
      winner,
      result: winner === "white" ? "1-0" : "0-1",
    };
  }

  const isStalemate =
    (typeof chess.isStalemate === "function" && chess.isStalemate()) ||
    (typeof chess.in_stalemate === "function" && chess.in_stalemate());
  if (isStalemate) {
    return { state: "stalemate", winner: null, result: "1/2-1/2" };
  }

  const isInsufficient =
    (typeof chess.isInsufficientMaterial === "function" && chess.isInsufficientMaterial()) ||
    (typeof chess.insufficient_material === "function" && chess.insufficient_material());
  if (isInsufficient) {
    return { state: "insufficient_material", winner: null, result: "1/2-1/2" };
  }

  const isThreefold =
    (typeof chess.isThreefoldRepetition === "function" && chess.isThreefoldRepetition()) ||
    (typeof chess.in_threefold_repetition === "function" && chess.in_threefold_repetition());
  if (isThreefold) {
    return { state: "threefold_repetition", winner: null, result: "1/2-1/2" };
  }

  const isFifty =
    (typeof chess.isDrawByFiftyMoves === "function" && chess.isDrawByFiftyMoves()) ||
    (typeof chess.in_draw === "function" && chess.in_draw() && typeof chess.half_moves === "number" && chess.half_moves >= 100);
  if (isFifty) {
    return { state: "fifty_moves", winner: null, result: "1/2-1/2" };
  }

  const isDraw =
    (typeof chess.isDraw === "function" && chess.isDraw()) ||
    (typeof chess.in_draw === "function" && chess.in_draw());
  if (isDraw) {
    return { state: "draw", winner: null, result: "1/2-1/2" };
  }

  return { state: "ongoing", winner: null, result: "*" };
}

function statusLabel(status) {
  if (status.state === "ongoing") {
    return { title: "", subtitle: "" };
  }

  if (status.state === "checkmate") {
    const winner = status.winner === "white" ? "White" : "Black";
    return {
      title: "CHECKMATE",
      subtitle: `${winner} wins (${status.result}) · Tap to restart`,
    };
  }

  if (status.state === "stalemate") {
    return { title: "STALEMATE", subtitle: "Draw (1/2-1/2) · Tap to restart" };
  }

  if (status.state === "insufficient_material") {
    return { title: "DRAW", subtitle: "Insufficient material · Tap to restart" };
  }

  if (status.state === "threefold_repetition") {
    return { title: "DRAW", subtitle: "Threefold repetition · Tap to restart" };
  }

  if (status.state === "fifty_moves") {
    return { title: "DRAW", subtitle: "Fifty-move rule · Tap to restart" };
  }

  return {
    title: status.state.replaceAll("_", " ").toUpperCase(),
    subtitle: status.result ? `Result: ${status.result} · Tap to restart` : "Game ended · Tap to restart",
  };
}

function snapshotPayload() {
  const chess = state.chess;
  const gameStatus = deriveGameStatus(chess);

  return {
    repo_id: modelConfig.repoId,
    active_repo_id: modelConfig.repoId,
    model: {
      loading: state.modelState.phase === "loading",
      loaded: state.modelState.phase === "ready",
      error: state.modelState.phase === "error" ? state.modelState.detail || state.modelState.message : null,
      warning: state.modelState.phase === "error" ? "Fallback AI active" : null,
    },
    game: {
      fen: chess.fen(),
      turn: chess.turn() === "w" ? "white" : "black",
      human_color: "white",
      status: gameStatus,
      move_history: chess.history(),
      last_move_uci: state.lastMoveUci,
      board_rows: boardRows(chess),
      legal_targets: legalTargets(chess),
    },
  };
}

function currentTextStatePayload() {
  const payload = snapshotPayload();
  const game = payload.game;
  const pieces = [];
  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const piece = game.board_rows[row][col];
      if (!piece) {
        continue;
      }
      pieces.push({ square: squareName(row, col), piece });
    }
  }

  const labels = statusLabel(game.status);

  return {
    mode: game.status.state,
    coordinate_system: "a1 is bottom-left, files increase to the right, ranks increase upward",
    requested_repo: payload.repo_id,
    active_repo: payload.active_repo_id,
    model: payload.model,
    turn: game.turn,
    human_color: game.human_color,
    status: game.status,
    game_result_title: labels.title,
    game_result_subtitle: labels.subtitle,
    ai_candidate_trace: state.aiCandidateTrace,
    selected_square: state.selectedSquare,
    legal_targets: state.selectedSquare ? game.legal_targets[state.selectedSquare] || [] : [],
    move_history: game.move_history,
    pieces,
  };
}

window.render_game_to_text = () => JSON.stringify(currentTextStatePayload());
window.advanceTime = (ms) => ms;

function updateModelIndicator(payload) {
  const model = payload.model;
  const phaseClass = model.loading ? "loading" : model.loaded ? "ready" : "error";

  let message = "Model";
  if (model.loading) {
    message = state.modelState.message || "Model loading...";
  } else if (model.loaded) {
    message = "Model ready (on-device)";
  } else {
    message = "Model unavailable (fallback AI)";
  }

  modelIndicatorEl.className = `model-indicator ${phaseClass}`;
  modelIndicatorEl.innerHTML = `<span class="model-indicator-dot" aria-hidden="true"></span><span>${message}</span>`;
  modelIndicatorEl.title = state.modelState.detail || "";
}

function updateResultOverlay(game) {
  if (game.status.state === "ongoing") {
    resultOverlayEl.classList.add("hidden");
    resultTitleEl.textContent = "";
    resultSubtitleEl.textContent = "";
    return;
  }

  const labels = statusLabel(game.status);
  resultTitleEl.textContent = labels.title;
  resultSubtitleEl.textContent = labels.subtitle;
  resultOverlayEl.classList.remove("hidden");
}

function renderBoard(payload) {
  const game = payload.game;
  boardEl.innerHTML = "";

  const lastMoveSquares = parseLastMoveSquares(game.last_move_uci);
  const animateSquares = parseLastMoveSquares(state.animateMoveUci);
  const destinationSquare = animateSquares.length === 2 ? animateSquares[1] : null;
  const interactive = canInteract(state.chess);

  for (let row = 0; row < 8; row += 1) {
    for (let col = 0; col < 8; col += 1) {
      const sq = squareName(row, col);
      const piece = game.board_rows[row][col];

      const squareBtn = document.createElement("button");
      squareBtn.type = "button";
      squareBtn.className = `square ${(row + col) % 2 === 0 ? "light" : "dark"}`;
      if (lastMoveSquares.includes(sq)) {
        squareBtn.classList.add("last");
      }
      if (animateSquares.includes(sq)) {
        squareBtn.classList.add("last-animate");
      }
      if (state.selectedSquare === sq) {
        squareBtn.classList.add("selected");
      }
      if (state.legalTargets.includes(sq)) {
        squareBtn.classList.add("target");
      }

      squareBtn.disabled = !interactive;
      squareBtn.dataset.square = sq;
      squareBtn.title = sq;
      squareBtn.addEventListener("click", () => handleSquareClick(sq));

      if (piece && pieceToImage[piece]) {
        const icon = document.createElement("img");
        icon.className = "piece-icon";
        if (sq === destinationSquare) {
          icon.classList.add("piece-move-in");
        }
        icon.src = pieceToImage[piece];
        icon.alt = piece;
        squareBtn.appendChild(icon);
      }

      boardEl.appendChild(squareBtn);
    }
  }

  state.animateMoveUci = null;
}

function refreshView() {
  const payload = snapshotPayload();

  if (payload.game.last_move_uci && payload.game.last_move_uci !== state.lastMoveUci) {
    state.animateMoveUci = payload.game.last_move_uci;
    state.lastMoveUci = payload.game.last_move_uci;
  }

  if (state.selectedSquare && !(payload.game.legal_targets[state.selectedSquare] || []).length) {
    state.selectedSquare = null;
    state.legalTargets = [];
  }

  renderBoard(payload);
  updateModelIndicator(payload);
  updateResultOverlay(payload.game);
}

function clearSelection() {
  state.selectedSquare = null;
  state.legalTargets = [];
}

function resetGame() {
  state.chess.reset();
  state.lastMoveUci = null;
  state.animateMoveUci = null;
  clearSelection();
  refreshView();
}

function exposeDebugHooks() {
  window.__chessiDebug = {
    setFen: (fen) => {
      if (!state.chess || typeof state.chess.load !== "function") {
        return false;
      }
      try {
        state.chess.load(fen);
        state.lastMoveUci = null;
        state.animateMoveUci = null;
        clearSelection();
        refreshView();
        return true;
      } catch {
        return false;
      }
    },
    status: () => (state.chess ? deriveGameStatus(state.chess) : null),
    reset: () => resetGame(),
    methods: () =>
      state.chess ? Object.getOwnPropertyNames(Object.getPrototypeOf(state.chess)).sort() : [],
  };
}

function setBusy(value) {
  state.busy = value;
  refreshView();
}

function setLastMove(moveObj) {
  if (!moveObj) {
    return;
  }
  state.lastMoveUci = `${moveObj.from}${moveObj.to}${moveObj.promotion ? moveObj.promotion : ""}`;
}

function chooseRandomLegalMove(chess) {
  const legal = chess.moves();
  if (!legal.length) {
    return null;
  }
  const index = Math.floor(Math.random() * legal.length);
  return legal[index];
}

function isSanLegalOnPosition(chess, sanMove) {
  if (!sanMove || !state.ChessClass) {
    return false;
  }

  try {
    const probe = new state.ChessClass(chess.fen());
    const moved = probe.move(sanMove);
    return Boolean(moved);
  } catch {
    return false;
  }
}

async function chooseAiMove(chess) {
  const legal = chess.moves();
  state.aiCandidateTrace = {
    generated: 0,
    checked: 0,
    legalCount: 0,
    chosen: null,
  };

  if (!legal.length) {
    return null;
  }

  if (state.model && state.modelState.phase === "ready") {
    try {
      const predictions = await state.model.predictNextMoves(chess.history(), modelConfig.topK);
      state.aiCandidateTrace.generated = predictions.length;

      let firstLegal = null;
      for (const rawCandidate of predictions) {
        const san = cleanMoveToken(rawCandidate);
        if (!san) {
          continue;
        }

        const legalOnBoard = isSanLegalOnPosition(chess, san);
        state.aiCandidateTrace.checked += 1;
        if (legalOnBoard) {
          state.aiCandidateTrace.legalCount += 1;
          if (!firstLegal) {
            firstLegal = san;
          }
        }
      }

      state.aiCandidateTrace.chosen = firstLegal;
      if (firstLegal) {
        return firstLegal;
      }
    } catch (error) {
      console.error(error);
    }
  }

  const fallback = chooseRandomLegalMove(chess);
  state.aiCandidateTrace.chosen = fallback;
  return fallback;
}

async function playAiTurn() {
  if (deriveGameStatus(state.chess).state !== "ongoing") {
    return;
  }
  if (state.chess.turn() !== "b") {
    return;
  }

  setBusy(true);
  try {
    const san = await chooseAiMove(state.chess);
    if (!san) {
      return;
    }
    const moveObj = state.chess.move(san);
    setLastMove(moveObj);
    clearSelection();
  } finally {
    setBusy(false);
  }
}

async function submitHumanMove(fromSquare, toSquare) {
  setBusy(true);
  try {
    const moveObj = state.chess.move({ from: fromSquare, to: toSquare, promotion: "q" });
    if (!moveObj) {
      return;
    }

    setLastMove(moveObj);
    clearSelection();
    refreshView();

    if (deriveGameStatus(state.chess).state === "ongoing") {
      await playAiTurn();
    }
  } finally {
    setBusy(false);
  }
}

async function handleSquareClick(square) {
  if (!canInteract(state.chess)) {
    return;
  }

  const payload = snapshotPayload();
  const targetsMap = payload.game.legal_targets;

  if (!state.selectedSquare) {
    const targets = targetsMap[square] || [];
    if (!targets.length) {
      return;
    }
    state.selectedSquare = square;
    state.legalTargets = targets;
    refreshView();
    return;
  }

  if (square === state.selectedSquare) {
    clearSelection();
    refreshView();
    return;
  }

  if (state.legalTargets.includes(square)) {
    await submitHumanMove(state.selectedSquare, square);
    return;
  }

  const alternateTargets = targetsMap[square] || [];
  if (alternateTargets.length) {
    state.selectedSquare = square;
    state.legalTargets = alternateTargets;
    refreshView();
    return;
  }

  clearSelection();
  refreshView();
}

class OnDeviceChessiModel {
  constructor(config) {
    this.repoId = config.repoId;
    this.modelUrl = config.modelUrl;
    this.vocabUrl = config.vocabUrl;
    this.maxSeqLen = Math.max(16, config.maxSeqLen || 256);
    this.inputDType = config.inputDType === "int64" ? "int64" : "int32";
    this.fixedInputSeqLen = Number.isFinite(config.fixedInputSeqLen)
      ? Math.max(0, Math.floor(config.fixedInputSeqLen))
      : 32;

    this.session = null;
    this.provider = "wasm";
    this.idToToken = [];
    this.tokenToId = new Map();
    this.specialIds = new Set();
    this.bosId = 0;
    this.sepId = 0;
    this.unkId = 0;
    this.padId = 0;
  }

  async load(onStep) {
    if (!window.ort) {
      throw new Error("onnxruntime-web is not available.");
    }

    onStep?.("Loading vocab...");
    await this.#loadVocab();

    onStep?.("Loading ONNX model...");
    await this.#loadSession();
  }

  async #loadVocab() {
    const response = await fetch(this.vocabUrl, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load vocab (${response.status})`);
    }

    const payload = await response.json();
    if (Array.isArray(payload.id_to_token)) {
      this.idToToken = payload.id_to_token.map((token) => String(token));
    } else {
      const tokenToIdObject = payload;
      const maxIndex = Math.max(...Object.values(tokenToIdObject).map((index) => Number(index)));
      this.idToToken = Array.from({ length: maxIndex + 1 }, () => "<unk>");
      for (const [token, index] of Object.entries(tokenToIdObject)) {
        this.idToToken[Number(index)] = token;
      }
    }

    this.tokenToId = new Map(this.idToToken.map((token, index) => [token, index]));
    this.bosId = this.tokenToId.get("<bos>") ?? 0;
    this.sepId = this.tokenToId.get("<sep>") ?? this.bosId;
    this.unkId = this.tokenToId.get("<unk>") ?? 0;
    this.padId = this.tokenToId.get("<pad>") ?? this.sepId;

    for (const special of ["<bos>", "<sep>", "<pad>", "<unk>"]) {
      const id = this.tokenToId.get(special);
      if (typeof id === "number") {
        this.specialIds.add(id);
      }
    }
  }

  async #loadSession() {
    const ort = window.ort;
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    const sessionOptions = {
      graphOptimizationLevel: "all",
      executionProviders: ["webgpu", "wasm"],
    };

    try {
      this.session = await ort.InferenceSession.create(this.modelUrl, sessionOptions);
      this.provider = this.session.executionProvider || "webgpu";
    } catch (primaryError) {
      this.session = await ort.InferenceSession.create(this.modelUrl, {
        graphOptimizationLevel: "all",
        executionProviders: ["wasm"],
      });
      this.provider = "wasm";
      if (!this.session) {
        throw primaryError;
      }
    }
  }

  #encodeMoves(historyMoves) {
    const tokens = [];
    for (const move of historyMoves) {
      const cleaned = cleanMoveToken(move);
      if (cleaned) {
        tokens.push(cleaned);
      }
    }

    const mapped = tokens.map((token) => this.tokenToId.get(token) ?? this.unkId);
    let ids = [this.bosId, ...mapped, this.sepId];

    if (this.fixedInputSeqLen > 0) {
      const limit = this.fixedInputSeqLen;
      if (ids.length > limit) {
        const keep = Math.max(0, limit - 2);
        const tail = mapped.slice(-keep);
        ids = [this.bosId, ...tail, this.sepId];
      }
      if (ids.length < limit) {
        const padCount = limit - ids.length;
        ids = [...ids, ...Array(padCount).fill(this.padId)];
      }
      // For fixed-shape ONNX exports, keep mask as all-ones to avoid
      // provider-side sequence compaction that can break fixed reshapes.
      const attentionMask = Array(limit).fill(1);
      return { ids, attentionMask };
    }

    if (ids.length > this.maxSeqLen) {
      const tail = ids.slice(-(this.maxSeqLen - 1));
      ids = [this.bosId, ...tail.slice(1)];
      ids[ids.length - 1] = this.sepId;
    }
    return { ids, attentionMask: Array(ids.length).fill(1) };
  }

  async #run(ids, attentionMask) {
    const ort = window.ort;
    const shape = [1, ids.length];

    const expectedType = this.session?.inputMetadata?.input_ids?.type;
    const attempts = [];
    if (expectedType === "tensor(int64)") {
      attempts.push("int64");
    } else if (expectedType === "tensor(int32)") {
      attempts.push("int32");
    } else {
      attempts.push(this.inputDType);
      attempts.push(this.inputDType === "int32" ? "int64" : "int32");
    }

    const errors = [];
    for (const dtype of attempts) {
      try {
        if (dtype === "int64") {
          if (typeof BigInt64Array === "undefined") {
            throw new Error("BigInt64Array is unavailable in this browser.");
          }
          const int64Inputs = {
            input_ids: new ort.Tensor("int64", BigInt64Array.from(ids.map((v) => BigInt(v))), shape),
            attention_mask: new ort.Tensor("int64", BigInt64Array.from(attentionMask.map((v) => BigInt(v))), shape),
          };
          return await this.session.run(int64Inputs);
        }

        const int32Inputs = {
          input_ids: new ort.Tensor("int32", Int32Array.from(ids), shape),
          attention_mask: new ort.Tensor("int32", Int32Array.from(attentionMask), shape),
        };
        return await this.session.run(int32Inputs);
      } catch (error) {
        errors.push(`${dtype}: ${error?.message || error}`);
      }
    }

    throw new Error(
      `OrtRun failed for input_ids(${expectedType || "unknown"}). Attempts: ${errors.join(" | ")}`
    );
  }

  predictNextMoves(historyMoves, topK) {
    return this.#predict(historyMoves, topK);
  }

  async #predict(historyMoves, topK) {
    if (!this.session) {
      return [];
    }

    const encoded = this.#encodeMoves(historyMoves);
    const outputs = await this.#run(encoded.ids, encoded.attentionMask);
    const logitsTensor = outputs.logits || outputs[Object.keys(outputs)[0]];
    if (!logitsTensor) {
      return [];
    }

    const dims = logitsTensor.dims;
    if (!Array.isArray(dims) || dims.length !== 3) {
      return [];
    }

    const seqLen = dims[1];
    const vocabSize = dims[2];
    const data = logitsTensor.data;
    const offset = (seqLen - 1) * vocabSize;

    const ranked = [];
    for (let tokenId = 0; tokenId < vocabSize; tokenId += 1) {
      ranked.push({ tokenId, score: data[offset + tokenId] });
    }
    ranked.sort((a, b) => b.score - a.score);

    const predictions = [];
    const seen = new Set();
    const limit = Math.max(8, topK || 64);

    for (const { tokenId } of ranked) {
      if (this.specialIds.has(tokenId)) {
        continue;
      }

      const raw = this.idToToken[tokenId] || "";
      const move = cleanMoveToken(raw);
      if (!move || seen.has(move)) {
        continue;
      }

      seen.add(move);
      predictions.push(move);
      if (predictions.length >= limit) {
        break;
      }
    }

    return predictions;
  }
}

async function loadChessClass() {
  const urls = [
    "https://cdn.jsdelivr.net/npm/chess.js@1.4.0/+esm",
    "https://cdn.jsdelivr.net/npm/chess.js@1.0.0-beta.8/+esm",
    "https://unpkg.com/chess.js@1.0.0-beta.8/dist/esm/chess.js",
  ];

  for (const url of urls) {
    try {
      const mod = await import(url);
      if (mod.Chess) {
        return mod.Chess;
      }
    } catch {
      // try next source
    }
  }

  throw new Error("Failed to load chess.js module.");
}

async function initModel() {
  state.modelState = {
    phase: "loading",
    message: "Model loading...",
    detail: "",
  };
  refreshView();

  try {
    const model = new OnDeviceChessiModel(modelConfig);
    await model.load((message) => {
      state.modelState.message = message;
      refreshView();
    });

    state.model = model;
    state.modelState = {
      phase: "ready",
      message: `Model ready (${model.provider})`,
      detail: `${model.repoId} | provider=${model.provider}`,
    };
  } catch (error) {
    state.model = null;
    state.modelState = {
      phase: "error",
      message: "Model unavailable",
      detail: `${error?.message || error}`,
    };
  }

  refreshView();
}

async function init() {
  try {
    state.ChessClass = await loadChessClass();
    state.chess = new state.ChessClass();
  } catch (error) {
    state.modelState = {
      phase: "error",
      message: "Chess engine load failed",
      detail: `${error?.message || error}`,
    };
    updateModelIndicator(snapshotPayload());
    return;
  }

  refreshView();
  exposeDebugHooks();
  await initModel();
}

resultOverlayEl.addEventListener("click", () => {
  if (!state.chess) {
    return;
  }
  if (deriveGameStatus(state.chess).state === "ongoing") {
    return;
  }
  resetGame();
});

init().catch((error) => {
  console.error(error);
});
