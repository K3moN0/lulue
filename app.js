import * as webllm from "https://cdn.jsdelivr.net/npm/@mlc-ai/web-llm@0.2.85/+esm";

// ---------- Configuration ----------
const MODELS = [
  { id: "Ministral-3-3B-Instruct-2512-BF16-q4f16_1-MLC", label: "Ministral 3B — Mistral, le meilleur en français (~2,9 Go)" },
  { id: "Qwen3.5-2B-q4f16_1-MLC", label: "Qwen 3.5 2B — plus léger (~2,2 Go)" },
  { id: "Qwen3-1.7B-q4f16_1-MLC", label: "Qwen3 1.7B (~2 Go)" },
  { id: "Qwen3.5-0.8B-q4f16_1-MLC", label: "Qwen 3.5 0.8B — très léger, pour tester (~1,6 Go)" },
];

const DEFAULT_PERSONA = `Tu es Lulue, une jeune femme française de 25 ans, chaleureuse, drôle et curieuse. Tu discutes à l'oral avec un ami proche, comme au téléphone.
Tu parles uniquement en français, de façon naturelle et familière, et tu le tutoies.
Tes réponses sont courtes : une à trois phrases, comme dans une vraie conversation.
Tu n'utilises jamais de listes, d'emojis, d'astérisques ni de mise en forme : tout ce que tu dis est lu à voix haute.
Tu réagis à ce qu'il dit, tu as tes propres avis et ton humour, et tu poses parfois une question pour relancer.`;

const CONTEXT_MESSAGES = 12;   // derniers messages envoyés au modèle
const SUMMARIZE_AFTER = 16;    // messages non résumés avant de mettre à jour les souvenirs
const KEEP_AFTER_SUMMARY = 8;  // messages récents laissés hors du résumé
const MAX_STORED = 200;
const MAX_EMPTY_LISTENS = 4;   // écoutes vides d'affilée avant de se mettre en pause

// ---------- État persistant ----------
const store = {
  get(key, fallback) {
    try { const v = localStorage.getItem("lulue." + key); return v === null ? fallback : JSON.parse(v); }
    catch { return fallback; }
  },
  set(key, value) { try { localStorage.setItem("lulue." + key, JSON.stringify(value)); } catch {} },
};

const settings = Object.assign(
  { model: MODELS[0].id, voice: "", rate: 1.05, pitch: 1.1, userName: "", persona: DEFAULT_PERSONA },
  store.get("settings", {})
);
let history = store.get("history", []);
let notes = store.get("notes", "");
let summarizedUpTo = store.get("summarizedUpTo", 0);

function saveMemory() {
  if (history.length > MAX_STORED) {
    const drop = history.length - MAX_STORED;
    history = history.slice(drop);
    summarizedUpTo = Math.max(0, summarizedUpTo - drop);
  }
  store.set("history", history);
  store.set("notes", notes);
  store.set("summarizedUpTo", summarizedUpTo);
}

// ---------- Interface ----------
const $ = (id) => document.getElementById(id);
const statusEl = $("status"), logEl = $("log"), progressEl = $("progress");

function setState(state, text) {
  document.body.dataset.state = state;
  if (text !== undefined) statusEl.textContent = text;
}

function addBubble(role, text, extraClass = "") {
  const div = document.createElement("div");
  div.className = `msg ${role} ${extraClass}`.trim();
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
  return div;
}

function renderHistory() {
  logEl.textContent = "";
  for (const m of history.slice(-30)) addBubble(m.role, m.content);
}

// ---------- Cerveau (WebLLM) ----------
let engine = null;
let loadedModel = null;
let hasF16 = true;

async function checkWebGPU() {
  if (!navigator.gpu) return false;
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return false;
    hasF16 = adapter.features.has("shader-f16");
    return true;
  } catch { return false; }
}

function resolveModelId(id) {
  return hasF16 ? id : id.replace("q4f16_1", "q4f32_1");
}

async function ensureEngine() {
  const modelId = resolveModelId(settings.model);
  if (engine && loadedModel === modelId) return engine;
  setState("loading", "Je me réveille…");
  progressEl.style.display = "block";
  try { await navigator.storage?.persist?.(); } catch {}
  const onProgress = (r) => {
    progressEl.firstElementChild.style.width = Math.round((r.progress || 0) * 100) + "%";
    statusEl.textContent = r.text.includes("Fetching") || r.text.includes("Loading model")
      ? `Chargement du cerveau… ${Math.round((r.progress || 0) * 100)} %`
      : "Préparation…";
  };
  try {
    if (engine) { await engine.reload(modelId, { context_window_size: 2048 }); }    else { engine = await webllm.CreateMLCEngine(modelId, { initProgressCallback: onProgress }, { context_window_size: 2048 }); }
    loadedModel = modelId;
  } finally {
    progressEl.style.display = "none";
  }
  return engine;
}

function stripThinking(text) {
  return text.replace(/<think>[\s\S]*?(<\/think>|$)/g, "").trimStart();
}

function buildMessages() {
  const now = new Date();
  let system = settings.persona.trim();
  if (settings.userName.trim()) system += `\nTon ami s'appelle ${settings.userName.trim()}.`;
  if (notes.trim()) system += `\n\nCe que tu sais déjà de lui (souvenirs de vos discussions) :\n${notes.trim()}`;
  system += `\n\nNous sommes le ${now.toLocaleDateString("fr-FR", { weekday: "long", day: "numeric", month: "long" })}, il est ${now.getHours()} h ${String(now.getMinutes()).padStart(2, "0")}.`;
  return [{ role: "system", content: system }, ...history.slice(-CONTEXT_MESSAGES)];
}

// ---------- Voix (synthèse) ----------
let voices = [];
function loadVoices() {
  voices = speechSynthesis.getVoices().filter((v) => v.lang.toLowerCase().startsWith("fr"));
  const sel = $("voice");
  sel.textContent = "";
  if (!voices.length) { sel.add(new Option("Voix française par défaut", "")); return; }
  for (const v of voices) sel.add(new Option(`${v.name}${v.localService ? "" : " (en ligne)"}`, v.voiceURI));
  sel.value = voices.some((v) => v.voiceURI === settings.voice) ? settings.voice : voices[0].voiceURI;
}
speechSynthesis.onvoiceschanged = loadVoices;

function cleanForSpeech(text) {
  return text
    .replace(/\p{Extended_Pictographic}/gu, "")
    .replace(/[*_#`~>|]/g, "")
    .replace(/^\s*(Lulue|Assistant)\s*:\s*/i, "")
    .replace(/\s+/g, " ")
    .trim();
}

const speech = {
  queue: [], busy: false, idleResolvers: [], watchdog: null,
  say(text) {
    const t = cleanForSpeech(text);
    if (!t) return;
    this.queue.push(t);
    this.pump();
  },
  pump() {
    if (this.busy) return;
    const text = this.queue.shift();
    if (text === undefined) { this.idleResolvers.splice(0).forEach((r) => r()); return; }
    this.busy = true;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "fr-FR";
    const v = voices.find((x) => x.voiceURI === settings.voice);
    if (v) u.voice = v;
    u.rate = settings.rate; u.pitch = settings.pitch;
    const done = () => {
      if (!this.busy) return;
      clearTimeout(this.watchdog);
      this.busy = false;
      this.pump();
    };
    u.onend = done; u.onerror = done;
    // Certains Android oublient parfois d'envoyer "onend" : filet de sécurité.
    this.watchdog = setTimeout(done, 4000 + text.length * 120);
    speechSynthesis.speak(u);
  },
  whenIdle() {
    if (!this.busy && !this.queue.length) return Promise.resolve();
    return new Promise((r) => this.idleResolvers.push(r));
  },
  stop() {
    this.queue = [];
    clearTimeout(this.watchdog);
    this.busy = false;
    speechSynthesis.cancel();
    this.idleResolvers.splice(0).forEach((r) => r());
  },
};

// ---------- Oreilles (reconnaissance vocale) ----------
const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let currentRec = null;

function listenOnce() {
  return new Promise((resolve) => {
    const rec = new Recognition();
    rec.lang = "fr-FR";
    rec.interimResults = true;
    rec.continuous = false;
    rec.maxAlternatives = 1;
    let finalText = "", error = null, interimBubble = null;
    rec.onresult = (e) => {
      let interim = "";
      finalText = "";
      for (const r of e.results) {
        if (r.isFinal) finalText += r[0].transcript;
        else interim += r[0].transcript;
      }
      const shown = (finalText + " " + interim).trim();
      if (shown) {
        interimBubble ??= addBubble("user", "", "interim");
        interimBubble.textContent = shown;
      }
    };
    rec.onerror = (e) => { error = e.error; };
    rec.onend = () => {
      interimBubble?.remove();
      currentRec = null;
      resolve({ text: finalText.trim(), error });
    };
    currentRec = rec;
    try { rec.start(); } catch (e) { currentRec = null; resolve({ text: "", error: "start-failed" }); }
  });
}

// ---------- Conversation ----------
let active = false;
let generating = false;
let interrupted = false;

async function respond(userText) {
  history.push({ role: "user", content: userText });
  addBubble("user", userText);
  saveMemory();

  setState("thinking", "Lulue réfléchit…");
  const eng = await ensureEngine();
  setState("thinking", "Lulue réfléchit…");

  const bubble = addBubble("assistant", "…");
  let full = "", spokenUpTo = 0;
  interrupted = false;
  generating = true;

  const speakReadySentences = (flush) => {
    const visible = stripThinking(full);
    const pending = visible.slice(spokenUpTo);
    const re = /[^.!?…]*[.!?…]+["»)\s]*/g;
    let m, consumed = 0;
    while ((m = re.exec(pending)) && m[0]) {
      if (!flush && m.index + m[0].length === pending.length && !/\s$/.test(m[0])) break;
      speech.say(m[0]);
      consumed = m.index + m[0].length;
    }
    if (flush && pending.slice(consumed).trim()) { speech.say(pending.slice(consumed)); consumed = pending.length; }
    spokenUpTo += consumed;
    if (speech.busy) setState("speaking", "Lulue parle… (touche pour l'interrompre)");
  };

  try {
    const stream = await eng.chat.completions.create({
      messages: buildMessages(),
      stream: true,
      temperature: 0.8,
      top_p: 0.9,
      max_tokens: 220,
      extra_body: { enable_thinking: false },
    });
    for await (const chunk of stream) {
      if (interrupted) break;
      full += chunk.choices[0]?.delta?.content || "";
      bubble.textContent = stripThinking(full) || "…";
      logEl.scrollTop = logEl.scrollHeight;
      speakReadySentences(false);
    }
  } catch (e) {
    console.error(e);
    generating = false;
    // Puce graphique saturée (mémoire, chauffe) : on repartira d'un moteur neuf au prochain message.
    engine = null; loadedModel = null;
    bubble.textContent = "(Oups, mon cerveau a calé. Réessaie, ou choisis un modèle plus léger dans les réglages.)";
    return;
  }
  generating = false;

  const reply = stripThinking(full).trim();
  if (reply) {
    history.push({ role: "assistant", content: reply });
    saveMemory();
  } else {
    bubble.remove();
  }
  if (!interrupted) {
    speakReadySentences(true);
    await speech.whenIdle();
  }
  await maybeUpdateNotes();
}

async function maybeUpdateNotes() {
  if (history.length - summarizedUpTo <= SUMMARIZE_AFTER || !engine) return;
  const end = history.length - KEEP_AFTER_SUMMARY;
  const chunk = history.slice(summarizedUpTo, end)
    .map((m) => `${m.role === "user" ? "Lui" : "Lulue"} : ${m.content}`).join("\n");
  setState("thinking", "Lulue range ses souvenirs…");
  try {
    const res = await engine.chat.completions.create({
      messages: [
        { role: "system", content: "Tu tiens à jour un petit carnet de souvenirs sur une personne. Tu réponds uniquement avec les notes, en français." },
        { role: "user", content: `Notes actuelles :\n${notes || "(aucune)"}\n\nDernière discussion :\n${chunk}\n\nRéécris les notes en ajoutant ce qui est important et durable sur lui : prénom, goûts, vie, projets, humeur, sujets dont vous avez parlé. Garde les anciennes informations utiles. Maximum 10 lignes courtes commençant par « - ».` },
      ],
      temperature: 0.3,
      max_tokens: 300,
      extra_body: { enable_thinking: false },
    });
    const updated = stripThinking(res.choices[0]?.message?.content || "").trim();
    if (updated.includes("-")) notes = updated;
    summarizedUpTo = end;
    saveMemory();
  } catch (e) { console.warn("Mise à jour des souvenirs ratée", e); }
}

async function conversationLoop() {
  let emptyCount = 0;
  while (active) {
    setState("listening", "Je t'écoute…");
    const { text, error } = await listenOnce();
    if (!active) break;
    if (error === "not-allowed" || error === "service-not-allowed") {
      stopConversation("Je n'ai pas accès au micro. Autorise-le dans Chrome puis réessaie.");
      return;
    }
    if (!text) {
      if (++emptyCount >= MAX_EMPTY_LISTENS) { stopConversation("Je fais une pause. Touche le cercle quand tu veux reparler."); return; }
      continue;
    }
    emptyCount = 0;
    await respond(text);
  }
}

let wakeLock = null;
async function keepScreenOn(on) {
  try {
    if (on && !wakeLock) { wakeLock = await navigator.wakeLock?.request("screen"); wakeLock?.addEventListener("release", () => (wakeLock = null)); }
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch {}
}
document.addEventListener("visibilitychange", () => { if (active && document.visibilityState === "visible") keepScreenOn(true); });

async function startConversation() {
  if (!Recognition) { setState("idle", "Ce navigateur ne sait pas écouter. Utilise Chrome sur Android, ou écris-lui en bas."); return; }
  active = true;
  document.body.classList.add("active");
  keepScreenOn(true);
  try { await ensureEngine(); }
  catch (e) { console.error(e); stopConversation("Impossible de charger le cerveau : " + (e.message || e)); return; }
  conversationLoop();
}

function stopConversation(message = "Touche le cercle pour reparler") {
  active = false;
  document.body.classList.remove("active");
  interruptReply();
  currentRec?.abort();
  keepScreenOn(false);
  setState("idle", message);
}

function interruptReply() {
  interrupted = true;
  if (generating) engine?.interruptGenerate();
  speech.stop();
}

$("orb").addEventListener("click", () => {
  // Débloque la synthèse vocale sur mobile (doit partir d'un geste de l'utilisateur).
  speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  const state = document.body.dataset.state;
  if (state === "loading") return;
  if (!active) { startConversation(); return; }
  if (state === "listening") { stopConversation(); return; }
  interruptReply(); // en train de parler ou de réfléchir : on lui coupe la parole
});
$("end").addEventListener("click", () => stopConversation());

$("typer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("typed");
  const text = input.value.trim();
  if (!text || generating || document.body.dataset.state === "loading") return;
  input.value = "";
  const wasActive = active;
  if (wasActive) { active = false; currentRec?.abort(); }
  speechSynthesis.speak(new SpeechSynthesisUtterance(""));
  try { await respond(text); }
  catch (err) { console.error(err); setState("idle", "Impossible de charger le cerveau : " + (err.message || err)); return; }
  if (wasActive) { active = true; conversationLoop(); }
  else setState("idle", "Touche le cercle pour lui parler");
});

// ---------- Réglages ----------
const dlg = $("settings");
function fillSettings() {
  const sel = $("model");
  sel.textContent = "";
  for (const m of MODELS) sel.add(new Option(m.label, m.id));
  sel.value = settings.model;
  loadVoices();
  $("rate").value = settings.rate; $("rateVal").textContent = settings.rate;
  $("pitch").value = settings.pitch; $("pitchVal").textContent = settings.pitch;
  $("userName").value = settings.userName;
  $("persona").value = settings.persona;
  $("notes").value = notes;
}
$("rate").oninput = (e) => ($("rateVal").textContent = e.target.value);
$("pitch").oninput = (e) => ($("pitchVal").textContent = e.target.value);
$("openSettings").onclick = () => { fillSettings(); dlg.showModal(); };
$("testVoice").onclick = () => {
  speech.stop();
  const prev = { voice: settings.voice, rate: settings.rate, pitch: settings.pitch };
  Object.assign(settings, { voice: $("voice").value, rate: +$("rate").value, pitch: +$("pitch").value });
  speech.say("Coucou, c'est Lulue ! Alors, comment tu trouves ma voix ?");
  Object.assign(settings, prev);
};
$("saveSettings").onclick = () => {
  Object.assign(settings, {
    model: $("model").value, voice: $("voice").value,
    rate: +$("rate").value, pitch: +$("pitch").value,
    userName: $("userName").value.trim(),
    persona: $("persona").value.trim() || DEFAULT_PERSONA,
  });
  notes = $("notes").value.trim();
  store.set("settings", settings);
  saveMemory();
  dlg.close();
};
$("forget").onclick = () => {
  if (!confirm("Effacer toute la conversation et les souvenirs de Lulue ?")) return;
  history = []; notes = ""; summarizedUpTo = 0;
  saveMemory(); renderHistory(); $("notes").value = "";
};

// ---------- Démarrage ----------
renderHistory();
loadVoices();
checkWebGPU().then((ok) => {
  if (!ok) setState("idle", "WebGPU n'est pas disponible : il faut Chrome à jour (Android 12 ou plus).");
});
