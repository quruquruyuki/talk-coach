// 会話コーチ（talk-coach）用 Edge Function
// ・transcribe：数秒ごとに区切った音声を文字にする（OpenAI）
// ・advise    ：これまでの会話を読んで「確認漏れ・あいまいな点・深掘りの質問・提案・注意」を返す（Claude）
// ・review    ：会話が終わったあとに、全体を振り返る（Claude）
// 音声・会話の中身はどこにも保存しない（保存は利用者の端末の中だけ）
//
// 使う Secrets（ほかのアプリと共用・設定済み）:
//   OPENAI_API_KEY    … 文字起こし
//   ANTHROPIC_API_KEY … 指摘と振り返り
//   ALLOWED_EMAIL     … このアプリを使えるログイン用メールアドレス
// モデルは共通設定（app_settings: talk-coach/transcribe_model, advise_model, review_model）から読む

const norm = (k: string) => k.toUpperCase().replace(/[^A-Z0-9]/g, "");
function env(name: string, ...aliases: string[]): string {
  const wanted = new Set([name, ...aliases].map(norm));
  for (const [k, v] of Object.entries(Deno.env.toObject())) {
    if (wanted.has(norm(k))) return v.trim().replace(/^["']|["']$/g, "").trim();
  }
  return "";
}
const OPENAI_API_KEY = env("OPENAI_API_KEY");
const ANTHROPIC_API_KEY = env("ANTHROPIC_API_KEY", "CLAUDE_API_KEY");
const ALLOWED = env("ALLOWED_EMAIL").toLowerCase().split(",").map((x) => x.trim()).filter(Boolean);
const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const APP = "talk-coach";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { ...CORS, "Content-Type": "application/json; charset=utf-8" } });
class HttpError extends Error { constructor(public status: number, message: string) { super(message); } }

// ── 共通設定（app_settings）──
let settingsCache: { at: number; map: Record<string, unknown> } | null = null;
async function settings(): Promise<Record<string, unknown>> {
  if (settingsCache && Date.now() - settingsCache.at < 60_000) return settingsCache.map;
  const map: Record<string, unknown> = {};
  try {
    const q = `select=app,key,value&app=in.${encodeURIComponent(`("global","${APP}")`)}`;
    const r = await fetch(`${SUPABASE_URL}/rest/v1/app_settings?${q}`, { headers: { apikey: SERVICE_KEY, Authorization: `Bearer ${SERVICE_KEY}` } });
    if (r.ok) for (const row of await r.json()) map[row.app === "global" ? `global.${row.key}` : row.key] = row.value;
  } catch (e) { console.error("app_settings", e); }
  settingsCache = { at: Date.now(), map };
  return map;
}
async function setting(key: string, fallback: string) {
  const v = (await settings())[key];
  return typeof v === "string" && v.trim() ? v.trim() : fallback;
}

// ── 料金（USD）──
const PRICE: Record<string, { in: number; out: number; ain?: number }> = {
  "claude-haiku-5-5": { in: 0.1, out: 0.5 },
  "claude-sonnet-5-5": { in: 2, out: 10 },
  "claude-opus-5-5": { in: 4, out: 20 },
};
const PER_MIN: Record<string, number> = { "gpt-transcribe": 0.0045, "whisper-1": 0.006 };
async function loadPrices() {
  const p = (await settings())["global.prices"];
  if (!p || typeof p !== "object") return;
  for (const [k, v] of Object.entries(p as Record<string, unknown>)) {
    if (k.endsWith("_per_min") && typeof v === "number") PER_MIN[k.slice(0, -8)] = v;
    else if (v && typeof v === "object" && typeof (v as { in?: unknown }).in === "number") PRICE[k] = v as { in: number; out: number };
  }
}
// deno-lint-ignore no-explicit-any
function claudeUsd(model: string, u: any) {
  const p = PRICE[model]; if (!p || !u) return null;
  const tin = (u.input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0) * 1.25 + (u.cache_read_input_tokens ?? 0) * 0.1;
  return (tin * p.in + (u.output_tokens ?? 0) * p.out) / 1e6;
}

// ── ログイン確認 ──
async function checkUser(req: Request) {
  if (!ALLOWED.length) throw new HttpError(500, "ALLOWED_EMAIL が設定されていません");
  const jwt = (req.headers.get("Authorization") ?? "").replace(/^Bearer\s+/i, "");
  if (!jwt) throw new HttpError(401, "ログインしてください");
  const r = await fetch(`${SUPABASE_URL}/auth/v1/user`, { headers: { Authorization: `Bearer ${jwt}`, apikey: SERVICE_KEY } });
  if (!r.ok) throw new HttpError(401, "ログインの期限が切れました。もう一度ログインしてください");
  const u = await r.json();
  if (!ALLOWED.includes(String(u.email ?? "").toLowerCase())) throw new HttpError(403, "このアカウントでは使えません");
}

// ── Claude ──
// deno-lint-ignore no-explicit-any
async function claude(model: string, system: any[], user: string, tool: any, maxTokens: number) {
  if (!ANTHROPIC_API_KEY) throw new HttpError(500, "ANTHROPIC_API_KEY が設定されていません");
  for (let i = 0; i < 3; i++) {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json" },
      body: JSON.stringify({
        model, max_tokens: maxTokens, system,
        tools: [tool], tool_choice: { type: "tool", name: tool.name },
        messages: [{ role: "user", content: user }],
      }),
    });
    if (r.ok) {
      const d = await r.json();
      // deno-lint-ignore no-explicit-any
      const t = (d.content ?? []).find((c: any) => c.type === "tool_use");
      if (!t) throw new HttpError(502, "AIの返事を読めませんでした");
      return { input: t.input, usd: claudeUsd(model, d.usage) };
    }
    const txt = await r.text();
    console.error("anthropic", model, r.status, txt.slice(0, 500));
    if (r.status === 429 || r.status >= 500) { await new Promise((s) => setTimeout(s, 800 * 2 ** i)); continue; }
    throw new HttpError(502, `AIエラー (${r.status}): ${txt.slice(0, 200)}`);
  }
  throw new HttpError(502, "AIが混み合っています");
}

// ── 文字起こし ──
async function transcribe(form: FormData) {
  if (!OPENAI_API_KEY) throw new HttpError(500, "OPENAI_API_KEY が設定されていません");
  const file = form.get("file") as File | null;
  if (!file || !file.size) throw new HttpError(400, "音声がありません");
  const model = await setting("transcribe_model", "gpt-transcribe");
  const prev = String(form.get("prev") ?? "").slice(-200);
  const hint = String(form.get("hint") ?? "").slice(0, 400);
  const fd = new FormData();
  fd.append("file", file, file.name || "a.m4a");
  fd.append("model", model);
  fd.append("language", "ja");
  fd.append("response_format", "json");
  const prompt = [hint && `出てくる可能性がある言葉: ${hint}`, prev && `直前の会話: ${prev}`].filter(Boolean).join("\n");
  if (prompt) fd.append("prompt", prompt);
  const r = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST", headers: { Authorization: `Bearer ${OPENAI_API_KEY}` }, body: fd,
  });
  const d = await r.json();
  if (!r.ok) throw new HttpError(502, `文字起こしエラー: ${d.error?.message ?? r.status}`);
  const sec = Number(form.get("seconds") ?? 0) || d.usage?.seconds || 0;
  let usd: number | null = null;
  if (PER_MIN[model] != null) usd = (sec / 60) * PER_MIN[model];
  else if (d.usage?.input_tokens != null && PRICE[model]) {
    const p = PRICE[model];
    usd = ((d.usage.input_token_details?.audio_tokens ?? d.usage.input_tokens) * (p.ain ?? p.in) + (d.usage.output_tokens ?? 0) * p.out) / 1e6;
  }
  let text = String(d.text ?? "").trim();
  // 無音のときに出がちな決まり文句は捨てる
  if (/^(ご視聴ありがとうございました|ありがとうございました|字幕|おやすみなさい)[。!！]?$/.test(text)) text = "";
  return { text, usd };
}

// ── 場面ごとの観点 ──
const MODES: Record<string, string> = {
  sell: `【場面：自分が提案・営業する側の商談】
観点：相手の課題・困りごとの具体（いつから、どれくらい困っているか、数字）、現状のやり方と不満、決める人・決め方（決裁者、ほかに関わる人）、予算感、時期・期限、比較している他社や代替案、判断の基準、懸念・反対意見の本音、次の約束（次回の日時・宿題）。
相手の発言の中で「それなら提案できる」というつながりを見つけたら offer で示す。`,
  buy: `【場面：自分が説明・提案を受ける側（住宅・保険・ローン・業者・見積もりなど）】
観点：総額と含まれないもの（別途費用・オプション）、追加費用が発生する条件、支払い時期と方法、解約・キャンセルの条件と費用、保証・アフター・期間、納期・工期と遅れた場合、口頭の約束を書面にしてもらえるか、他社との違い、デメリットや向いていない人、「標準」「サービス」「大体」などあいまいな言葉の中身。
相手が良いことだけを言っているときは、反対側（デメリット・例外）を聞く質問を出す。`,
  mr: `【場面：製薬会社のMRが医師と面談している】
観点：先生の関心・困っている患者像、現在の処方・使い分け、使用経験と評価、懸念点（効果・副作用・飲みにくさ・コスト）、資料や情報提供の依頼、勉強会・説明会の希望、次回の約束。
先生の発言に、薬を使った患者に起きた好ましくない症状・検査値の変化が含まれたら、必ず warn で「副作用の可能性：詳細（患者背景・時期・転帰）の確認」を出す。
適応外の使い方を勧める・効果を言い過ぎるなど、MR側の発言に問題がありそうなら warn で知らせる。`,
  meet: `【場面：打ち合わせ・相談】
観点：目的とゴールの確認、決まったこと・決まっていないこと、担当と期限、前提条件のずれ、数字や固有名詞のあいまいさ、反対意見や懸念、次にやること。`,
};

const SYSTEM_BASE = `あなたは、会話を横で聞いている優秀な先輩です。利用者（自分）がいま会話をしている最中で、スマホの画面をちらっと見るだけで役に立つ一言を出します。
出すものは5種類：
- missing：まだ聞けていない・確認すべきなのに出ていないこと
- unclear：あいまいな言葉、数字の食い違い、前の発言との矛盾、根拠のない断定
- ask：いま聞くと話が深まる質問（そのまま口に出せる言い方で）
- offer：相手の話とつながる提案・切り返し
- warn：注意（言い過ぎ・約束しすぎ・不利な条件・副作用の情報など）
ルール：
- 本当に役立つものだけ、多くても3つ。とくになければ cards は空でよい。毎回何かを出そうとしない。
- すでに出した指摘（prior）と同じ内容は出さない。言い方を変えただけのものも出さない。
- 会話で実際に出た内容に基づく。話していないことを話したことにしない。
- 文字起こしには聞き間違いがある。おかしな言葉は文脈から読み替える。誰の発言かは書かれていないので、内容から判断する。
- text は画面で一瞬で読める短さ（全角30字以内が目安）。ask は相手にそのまま言える丁寧な言い方。
- why には、なぜそれを出したかを一言（全角40字以内）。会話の中の根拠の発言があれば短く添える。
- 出力はすべて日本語。`;

const ADVISE_TOOL = {
  name: "coach",
  description: "会話中の利用者に見せる一言と、確認リストの状況を返す",
  input_schema: {
    type: "object",
    properties: {
      cards: {
        type: "array", maxItems: 3,
        items: {
          type: "object",
          properties: {
            type: { type: "string", enum: ["missing", "unclear", "ask", "offer", "warn"] },
            text: { type: "string" },
            why: { type: "string" },
          },
          required: ["type", "text"],
        },
      },
      covered: { type: "array", items: { type: "string" }, description: "確認リストのうち、会話の中で答えが出た・確認できた項目のid" },
      summary: { type: "string", description: "fold が true のときだけ：これまでの要約＋今回の会話を合わせた、新しい要約（600字以内。数字・固有名詞・決まったこと・相手の懸念を優先して残す）。fold が false なら空" },
    },
    required: ["cards", "covered"],
  },
};

// deno-lint-ignore no-explicit-any
function setupText(s: any) {
  const list = (Array.isArray(s?.checklist) ? s.checklist : []).slice(0, 40)
    // deno-lint-ignore no-explicit-any
    .map((c: any) => `- id=${String(c.id).slice(0, 12)}：${String(c.text ?? "").slice(0, 120)}`).join("\n");
  return [
    MODES[s?.mode] ?? MODES.meet,
    s?.goal ? `【今日の目的】${String(s.goal).slice(0, 500)}` : "",
    s?.counterpart ? `【相手】${String(s.counterpart).slice(0, 200)}` : "",
    list ? `【確認リスト】\n${list}` : "",
    s?.offers ? `【自分の手持ちの提案・材料】\n${String(s.offers).slice(0, 1500)}` : "",
    s?.notes ? `【メモ・前提】\n${String(s.notes).slice(0, 1500)}` : "",
  ].filter(Boolean).join("\n\n");
}

// deno-lint-ignore no-explicit-any
async function advise(body: any) {
  const recent = String(body.recent ?? "").slice(-6000);
  if (!recent.trim()) return { cards: [], covered: [], summary: "", usd: 0 };
  const model = await setting("advise_model", "claude-haiku-5-5");
  const fold = !!body.fold;
  // 会話中は変わらない部分（指示・場面・確認リスト）はキャッシュして安くする
  const system = [{ type: "text", text: `${SYSTEM_BASE}\n\n${setupText(body.setup)}`, cache_control: { type: "ephemeral" } }];
  const prior = (Array.isArray(body.prior) ? body.prior : []).slice(-20).map((x: unknown) => `- ${String(x).slice(0, 80)}`).join("\n");
  const open = (Array.isArray(body.open) ? body.open : []).slice(0, 40).join(", ");
  const user = [
    `経過時間：約${Math.round(Number(body.elapsed_min) || 0)}分`,
    body.summary ? `【これまでの要約】\n${String(body.summary).slice(0, 2000)}` : "",
    `【直近の会話（文字起こし）】\n${recent}`,
    prior ? `【すでに出した指摘（prior）】\n${prior}` : "",
    open ? `【確認リストでまだ済んでいない項目のid】${open}` : "",
    body.ending ? "※会話が終わりに近づいています。次の約束・宿題・決まっていないことを優先してください。" : "",
    `fold=${fold}`,
  ].filter(Boolean).join("\n\n");
  const { input, usd } = await claude(model, system, user, ADVISE_TOOL, 900);
  const types = new Set(["missing", "unclear", "ask", "offer", "warn"]);
  // deno-lint-ignore no-explicit-any
  const cards = (Array.isArray(input.cards) ? input.cards : []).filter((c: any) => types.has(c?.type) && c?.text)
    // deno-lint-ignore no-explicit-any
    .slice(0, 3).map((c: any) => ({ type: c.type, text: String(c.text).slice(0, 80), why: String(c.why ?? "").slice(0, 120) }));
  const covered = (Array.isArray(input.covered) ? input.covered : []).map(String).slice(0, 40);
  return { cards, covered, summary: fold ? String(input.summary ?? "").slice(0, 2500) : "", usd };
}

// ── 振り返り ──
const REVIEW_TOOL = {
  name: "review",
  description: "会話全体の振り返り",
  input_schema: {
    type: "object",
    properties: {
      title: { type: "string", description: "一覧で見分けられる短いタイトル" },
      summary: { type: "string", description: "3〜5文の要約" },
      decided: { type: "array", items: { type: "string" }, description: "決まったこと" },
      missed: { type: "array", items: { type: "string" }, description: "最後まで確認できなかったこと・聞きそびれたこと（重要な順）" },
      unclear: { type: "array", items: { type: "string" }, description: "あいまいなまま・食い違ったままのこと" },
      next: { type: "array", items: { type: "string" }, description: "次にやること・相手への確認事項（担当がわかれば添える）" },
      good: { type: "array", items: { type: "string" }, description: "うまくできていた点（1〜3個）" },
      improve: { type: "array", items: { type: "string" }, description: "次の会話で改善できる点（1〜3個）" },
      followup: { type: "string", description: "相手に送るお礼・確認のメッセージの下書き（丁寧語、200〜400字）。不要な場面なら空" },
    },
    required: ["title", "summary", "missed", "next"],
  },
};
// deno-lint-ignore no-explicit-any
async function review(body: any) {
  const transcript = String(body.transcript ?? "").slice(-60000);
  if (!transcript.trim()) throw new HttpError(400, "会話の文字起こしがありません");
  const model = await setting("review_model", "claude-sonnet-5-5");
  const system = [{ type: "text", text: `あなたは会話の振り返りを手伝う先輩です。文字起こしには聞き間違いがあるので文脈から読み替え、話していないことは書かないでください。出力は日本語。\n\n${setupText(body.setup)}` }];
  const shown = (Array.isArray(body.cards) ? body.cards : []).slice(-40).map((x: unknown) => `- ${String(x).slice(0, 100)}`).join("\n");
  const user = [`【会話の文字起こし】\n${transcript}`, shown ? `【会話中に画面に出した指摘】\n${shown}` : ""].filter(Boolean).join("\n\n");
  const { input, usd } = await claude(model, system, user, REVIEW_TOOL, 4000);
  return { result: input, usd };
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response("ok", { headers: CORS });
  if (req.method !== "POST") return json({ error: "POST only" }, 405);
  try {
    await checkUser(req);
    await loadPrices();
    if ((req.headers.get("content-type") ?? "").includes("multipart/form-data")) {
      return json(await transcribe(await req.formData()));
    }
    // deno-lint-ignore no-explicit-any
    const body: any = await req.json().catch(() => ({}));
    if (body.action === "advise") return json(await advise(body));
    if (body.action === "review") return json(await review(body));
    if (body.action === "ping") return json({ ok: true, rate: Number((await settings())["global.usd_jpy"]) || 150 });
    return json({ error: "unknown action" }, 400);
  } catch (e) {
    const status = e instanceof HttpError ? e.status : 500;
    console.error(e);
    return json({ error: e instanceof Error ? e.message : String(e) }, status);
  }
});
