/**
 * KNL07 Specification & Factory (single-file)
 * Index / Navigation
 *   Core Stack (search `/* === C-`)
 *     C-0 Flags/Metrics — runtime toggles・観測値。
 *     C-1 SymbolClass — DSLトークン定義とベースクラス。
 *     C-2 Tokenizer — 正規表現＋先頭文字ディスパッチ、インクリメンタル対応。
 *     C-3 Parser — NodeSet/ExecutionBlock生成、from/to・I/O解決。
 *     C-4 Knowledge — 辞書マージと属性補強。
 *     C-5 Prompt Builder — Persona/出力制約付きプロンプト生成。
 *     C-6 Plans — ^?計画、要件検証、DSL採用ユーティリティ。
 *     C-7 Runtime — 実行ダグ、LLMクライアント、変数解析、APIアクセス。
 *   Facade (`KNL07Core`) — tokenize/parse/enrich/prompt/runtime を束ねるファクトリ。
 *   UI Stack (search `/* === U-`)
 *     U-1 Header/AIAssist/Tokenizer
 *     U-2 Tree Viewer
 *     U-3 Workflow SVG
 *     U-4 Swimlane Builder
 *     U-5 Executor & History
 *     U-6 Theme / Global Memory utilities
 * 
 * Legend
 *   ノード: `^` / `^?`、属性: " @ =$ ($) # × ; ✓ (!)、演算子: | & > § ↓。
 *   Spec Capsules (// [Spec] C-*) summarize責務、IO、依存関係。
 *
 * Quick use case / Aims（簡易ユースケースとねらい）
 *   - 記号ベースの Workflow DSL で LLM タスクを「分割実行」するための表現言語。
 *     原理としては^task | ^task2 のように[^記号⁺コンテキスト(node)]+演算子記号+[^記号⁺コンテキスト(node)]の繰り返しで誰でも思考処理を表現できることを目指す。
 *     発想としてはメモの箇条書きやWBSに近く、文字ベースの強みは「編集のしやすさ」「バージョン管理のしやすさ」「差分把握のしやすさ」にある。
 *       例: `^調査A & ^調査B ↓ ^要約 | ^レビュー✓` // 並列調査→要約→レビュー
 *   - LLM 自身にも出力しやすい簡易表記で、LLM にとっての「意思実行/自律思考言語」の狙いから作成している。意志符号^と記憶符号$と実行計画系[|&>§↓]、その他修飾[@persona,;,×...]。
 *       例: モデルに「この要件で DSL を生成して」と促すと、そのまま採用可能な DSL を返す。
 *        //C6^?はこのための自律計画＆実行ノード。U1のDSL生成支援機能は思考計画。辞書は外部装置
 *   - オマケとして、人間の思考処理の言語化・可視化（可視化タブ/スイムレーン）を支援。
 *       Mermaid / Notion などと用途が一部重なるが、本質は「実行可能 DSL」と「分割実行」にある。
 *
 *   DSL ミニ例:
 *     ^リサーチA ($topic) & ^リサーチB ($topic) ↓ ^要約@編集者 =!summary | ^レビュー✓
 *     ^?リサーチ計画 ($topic) ↓ ^要約@編集者 =!summary | ^レビュー✓
 *     ^アイディア出し=$idealist | ^選択
 *     ^taskA & ^taskB ↓ ^check
 *     //if/then/else は^evaluateや=$最適案で表現する。より厳密で複雑な機械処理が欲しい場合、関数呼び出しを検討。
 * 
 *思想とコードの離散・肥大抑止のためあえてシングルファイル化。これで崩壊するならそこまで。中核原理は圧縮、単純、公理化してなんぼ。
 */
import React from "react";
// ELK (layered graph layout) for better crossing minimization
// eslint-disable-next-line @typescript-eslint/ban-ts-comment
// @ts-ignore - elkjs has no types shipped by default
import ELK from "elkjs/lib/elk.bundled.js";

// [Spec] C-0 Flags / Metrics / Utils
// Purpose: 集中管理されたフラグ・メトリクス・ユーティリティで、他セクションが副作用/挙動を切り替えるために参照。
// Provides: `KNL07Flags`, `KNL07Metrics`, 時刻/文字列整形ヘルパー (`nowIso`, `safeTruncate`).
// Consumers: Tokenizer(C-2), Parser(C-3), Runtime(C-7), UIログなどが依存。
// Mutation policy: Flagsは実行時に変更可能だが初期値はここで定義、Metricsは集計用途。

/* ================================================================
 * C-0: Flags / Metrics / Common Utils
 * ================================================================ */
export const KNL07Flags = {
  tokenizerFirstCharDispatch: true,
  llmRetry: true,
  llmTimeoutMs: 45_000,
  llmMaxRetries: 2,
  llmInitialBackoffMs: 1_200,
  llmJitter: true,

  limitHistoryTail: 12,
  limitContextSnapshot: 12,

  summarizeExcludeNodeKeys: true,
  summarizeExcludeResultKeys: true,

  // 追加: 依存系譜の会話履歴を LLM に渡す
  useLineageHistory: true,
  lineageMaxPairs: 50, // 先祖タスクの最大(プロンプト+レスポンス)組数
} as const;

export const KNL07Metrics = {
  counters: {} as Record<string, number>,
  incr(k: string, n = 1) {
    this.counters[k] = (this.counters[k] ?? 0) + n;
  },
  dump() {
    // eslint-disable-next-line no-console
    console.table(this.counters);
  },
};

export function nowIso(): string {
  return new Date().toISOString();
}

export function safeTruncate(value: unknown, max = 140): string {
  let str: string | null = null;
  if (typeof value === "string") str = value;
  else {
    try {
      str = JSON.stringify(value);
    } catch {
      str = String(value ?? "");
    }
  }
  if (!str) return "";
  return str.length > max ? `${str.slice(0, max)}…` : str;
}

// [Spec] C-1 / C-2 Token Foundations
// Purpose: 定義済み DSL 記号をクラスとして表現し、トークン生成/適用ポリシーを一元管理。
// Provides: `KNLToken`/`NodeToken` ベースクラス、属性/演算子/ノード各種ファクトリ、共通正規表現定数、`makeStableNodeGuid` など。
// Consumers: Tokenizer(C-2), Parser(C-3), Knowledge(C-4)、UIノード表示で型チェックに使用。
// Extension: 新しい記号を追加する場合はここにクラスを定義し、`KNL07_TOKEN_CLASSES` 登録を更新。

/* ================================================================
 * C-1/C-2: Token Types / Base Classes / Common regex constants
 * ================================================================ */
export type AttrPolicy = "lastOnly" | "primaryOnly" | "setAll";

export interface KNL07TokenDocEntry {
  type: string;
  symbol: string;
  description: string;
  placement?: "before" | "after";
  example?: string;
  notes?: string;
}
export type KNL07TokenDocConfig = Partial<Omit<KNL07TokenDocEntry, "type" | "symbol">> & {
  description?: string;
  symbol?: string;
};

export interface KNL07TokenConstructor<T extends KNLToken = KNLToken> {
  new (...args: any[]): T;
  type: string;
  symbol?: string;
  regex: RegExp;
  priority: number;
  description: string;
  doc?: KNL07TokenDocConfig;
}

/* 共通文字クラス（ver5: 定数化） */
const JP_ID = "[\\w\\.\\s\\u3000-\\u33FF\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF\\uFF66-\\uFF9F]+";
const VAR_ID = "[a-zA-Z0-9_\\u3000-\\u33FF\\u3400-\\u4DBF\\u4E00-\\u9FFF\\uF900-\\uFAFF\\uFF66-\\uFF9F]+";
const NON_TOKEN_CLASS = "[^\\^\\$\\?|&>↓§;✓\"@=\\(\\)!:\\/\\.\\×#\\s]+";

export abstract class KNLToken {
  static symbol: string;
  static regex: RegExp;
  static type: string;
  static priority: number;
  static description: string;
  static doc?: KNL07TokenDocConfig;

  id: string;
  type: string;
  value: string;
  position: number;
  description: string;
  attributes: Record<string, any>;
  tokenIndex?: number;

  constructor(type: string, value: string, position: number = 0, description: string = "") {
    this.id = `${type}-${position}`;
    this.type = type;
    this.value = value.trim();
    this.position = position;
    this.description = description;
    this.attributes = {};
  }
}

export abstract class NodeToken extends KNLToken {
  nodeindex?: number;
  typeIndex?: number;
  guid: string = "";
  from: number[] = [];
  to: number[] = [];
  explicitInputs: string[] = [];
  explicitOutputs: string[] = [];
  explicitRequirements: string[] = [];
  calculatedInputs: { fromNode: number; input: string }[] = [];
  dsl: string = "";
  dslStart?: number;
  dslEnd?: number;
  dslRaw: string = "";
}

export function makeStableNodeGuid(type: string, value: string, position: number, index: number): string {
  const base = `${type}:${value}:${position}:${index}`;
  let hash = 0;
  for (let i = 0; i < base.length; i++) {
    hash = (hash << 5) - hash + base.charCodeAt(i);
    hash |= 0;
  }
  const normalized = Math.abs(hash).toString(36);
  return `node-${normalized}`;
}

export type NodeInputRecord = { input: string; fromNode: number | undefined };

export function getNodeInputRecords(node: NodeToken): NodeInputRecord[] {
  if (node.calculatedInputs && node.calculatedInputs.length) {
    return node.calculatedInputs.map((ci) => ({
      input: ci.input,
      fromNode: typeof ci.fromNode === "number" && !Number.isNaN(ci.fromNode) ? ci.fromNode : undefined,
    }));
  }
  return (node.explicitInputs ?? []).map((input) => ({ input, fromNode: undefined }));
}

/* AttrToken 共通: 対象解決のヘルパー（ver5: 追加） */
function selectTargetIndices(
  policy: AttrPolicy,
  primaryIdx: number,
  nodeListIdxs: number[],
  nodeIndices: number[],
): number[] {
  if (policy === "primaryOnly") return [primaryIdx];
  if (policy === "setAll") return nodeListIdxs;
  const last = nodeIndices[nodeIndices.length - 1];
  return typeof last === "number" ? [last] : [];
}

function applyToTargets(targets: number[], nodes: NodeToken[], fn: (n: NodeToken) => void): void {
  targets.forEach((idx) => fn(nodes[idx]));
}

export abstract class AttrToken extends KNLToken {
  placement: "before" | "after";
  constructor(type: string, value: string, pos: number, desc: string, place: "before" | "after") {
    super(type, value, pos, desc);
    this.placement = place;
  }
  protected resolveTargets(
    nodes: NodeToken[],
    nodeIndices: number[],
    policy: AttrPolicy,
    primaryIdx: number,
    nodeListIdxs: number[],
  ): number[] {
    return selectTargetIndices(policy, primaryIdx, nodeListIdxs, nodeIndices);
  }
  abstract assignAttr(
    nodes: NodeToken[],
    nodeIndices: number[],
    policy: AttrPolicy,
    primaryIdx: number,
    nodeListIdxs: number[],
  ): void;
}

export abstract class OperatorToken extends KNLToken {
  abstract layout: "row" | "column";
  abbreviation?: string;
}

/* ================================================================
 * C-2: Mini Factories + Tokens + Registry + Tokenizer
 * ================================================================ */
type AttrApply = (params: {
  token: AttrToken;
  nodes: NodeToken[];
  nodeIndices: number[];
  policy: AttrPolicy;
  primaryIdx: number;
  nodeListIdxs: number[];
}) => void;

function createAttrTokenClass(spec: {
  type: string;
  regex: RegExp;
  priority: number;
  description: string;
  placement: "before" | "after";
  symbol?: string;
  doc?: KNL07TokenDocConfig;
  apply: AttrApply;
}) {
  const { type, regex, priority, description, placement, symbol, doc, apply } = spec;
  return class GeneratedAttrToken extends AttrToken {
    static symbol = symbol ?? "";
    static regex = regex;
    static type = type;
    static priority = priority;
    static description = description;
    static doc = doc;
    constructor(v: string, p = 0) {
      super(type, v, p, description, placement);
    }
    assignAttr(nodes: NodeToken[], nodeIndices: number[], policy: AttrPolicy, primaryIdx: number, nodeListIdxs: number[]) {
      const targets = this.resolveTargets(nodes, nodeIndices, policy, primaryIdx, nodeListIdxs);
      apply({ token: this, nodes, nodeIndices: targets, policy, primaryIdx, nodeListIdxs });
    }
  };
}

function createOperatorTokenClass(spec: {
  type: string;
  symbol: string;
  regex: RegExp;
  priority: number;
  description: string;
  abbreviation?: string;
  layout: "row" | "column";
  doc?: KNL07TokenDocConfig;
}) {
  const { type, symbol, regex, priority, description, abbreviation, layout, doc } = spec;
  return class GeneratedOperatorToken extends OperatorToken {
    static symbol = symbol;
    static regex = regex;
    static type = type;
    static priority = priority;
    static description = description;
    static doc = doc;
    abbreviation?: string = abbreviation;
    layout: "row" | "column" = layout;
    constructor(v: string, p = 0) {
      super(type, v, p, description);
    }
  };
}

function createNodeTokenClass(spec: {
  type: string;
  symbol: string;
  regex: RegExp;
  priority: number;
  description: string;
  doc?: KNL07TokenDocConfig;
  base?: typeof NodeToken;
}) {
  const { type, symbol, regex, priority, description, doc, base } = spec;
  const Base = base ?? NodeToken;
  return class GeneratedNodeToken extends Base {
    static symbol = symbol;
    static regex = regex;
    static type = type;
    static priority = priority;
    static description = description;
    static doc = doc;
    constructor(v: string, p = 0) {
      super(type, v, p, description);
    }
  };
}

/* Node tokens */
export const TaskToken = createNodeTokenClass({
  type: "TASK",
  symbol: "^",
  regex: new RegExp(`^\\^${JP_ID}`),
  priority: 100,
  description: "タスク",
  doc: { description: "具体的なタスクノード。", example: "^競合調査" },
}) as unknown as { new (v: string, p?: number): NodeToken } & typeof NodeToken;

export const UndecidedTaskToken = class extends (TaskToken as any) {
  static symbol = "^?";
  static regex = new RegExp(`^\\^\\?${JP_ID}`);
  static type = "UNDECIDED_TASK";
  static priority = 110;
  static description = "未定タスク";
  static doc = { description: "^? で始まる未定タスク。Plan ジェネレータがプラン候補を生成。", example: "^?追加調査案" };
  constructor(v: string, p = 0) {
    super(v, p);
    (this as any).type = (UndecidedTaskToken as any).type;
    this.description = (UndecidedTaskToken as any).description;
  }
} as unknown as { new (v: string, p?: number): NodeToken } & typeof NodeToken;

export const VariableNodeToken = createNodeTokenClass({
  type: "VARIABLE",
  symbol: "$",
  regex: new RegExp(`^\\$${VAR_ID}`),
  priority: 100,
  description: "変数ノード",
  doc: { description: "DSL 内変数参照。プランやパーサで値を束縛する。", example: "$marketData" },
}) as unknown as { new (v: string, p?: number): NodeToken } & typeof NodeToken;

export const PersonaNodeToken = createNodeTokenClass({
  type: "PERSONA_NODE",
  symbol: "@",
  regex: /^@[\w\u3000-\u33FF\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF\uFF66-\uFF9F]+/,
  priority: 120,
  description: "ペルソナ",
  doc: { description: "実行主体を表すペルソナノード。タスクにアタッチする", example: "^UIを改善@デザイナー" },
}) as unknown as { new (v: string, p?: number): NodeToken } & typeof NodeToken;

/* Attr tokens */
export const DescriptionToken = createAttrTokenClass({
  type: "DESCRIPTION",
  // 変更: \" を許容
  regex: /^"(?:[^"\\]|\\.)*"/,
  priority: 90,
  description: "説明",
  placement: "after",
  symbol: '"',
  apply: ({ token, nodes, nodeIndices }) => {
    // 外側の "..." を剥がし、エスケープを戻す
    const raw = token.value.slice(1, -1);
    const d = raw.replace(/\\"/g, '"').replace(/\\\\/g, '\\');
    applyToTargets(nodeIndices, nodes, (n) => (n.attributes["description"] = d));
  },
});

export const ExplicitInputToken = createAttrTokenClass({
  type: "EXPLICIT_INPUT",
  regex: new RegExp(`^\\(\\$${VAR_ID}\\)`),
  priority: 85,
  description: "明示入力",
  placement: "after",
  doc: { symbol: "($…)", description: "ノードへ明示的に流し込む入力変数を指定。", example: "^翻訳 ($draftCopy)" },
  apply: ({ token, nodes, nodeIndices }) => {
    const v = token.value.slice(1, -1);
    applyToTargets(nodeIndices, nodes, (n) => n.explicitInputs.push(v));
  },
});

export const ExplicitOutputToken = createAttrTokenClass({
  type: "EXPLICIT_OUTPUT",
  regex: new RegExp(`^=\\$${VAR_ID}(\\[\\]|\\[\\d+\\])?`),
  priority: 85,
  description: "明示出力（a, a[], a[1]）",
  placement: "after",
  doc: { symbol: "=$…", description: "ノードの出力を変数へ束縛。配列/添字記法対応。", example: "^下調べ =$research[0]" },
  apply: ({ token, nodes, nodeIndices }) => {
    const out = token.value.substring(1);
    applyToTargets(nodeIndices, nodes, (n) => n.explicitOutputs.push(out));
  },
});

export const RequirementInputToken = createAttrTokenClass({
  type: "REQUIREMENT_INPUT",
  regex: new RegExp(`^\\(!${VAR_ID}\\)`),
  priority: 85,
  description: "要件入力",
  placement: "after",
  symbol: "(!)",
  doc: { description: "要件として必須な入力変数を宣言。欠落チェックに利用。", example: "^集計 (!sourceData)" },
  apply: ({ token, nodes, nodeIndices }) => {
    const v = token.value.slice(2, -1);
    applyToTargets(nodeIndices, nodes, (n) => n.explicitRequirements.push(v));
  },
});

export const OutputTypeSpecToken = createAttrTokenClass({
  type: "OUTPUT_TYPE_SPEC",
  regex: /^=![A-Za-z0-9_]+(?:\/![A-Za-z0-9_]+)*(?:\/\.)?/,
  priority: 85,
  description: "出力型分岐",
  placement: "after",
  symbol: "=!",
  doc: { description: "想定する出力タイプを宣言し、プロンプト制約と検証に反映。", example: "^分析 =!report/!summary/." },
  apply: ({ token, nodes, nodeIndices }) => {
    const raw = token.value.slice(2);
    const parts = raw.split("/");
    const types: string[] = [];
    let allowOther = false;
    for (const p of parts) {
      if (p === ".") {
        allowOther = true;
        continue;
      }
      const t = p.replace(/^!/, "").trim();
      if (t) types.push(t);
    }
    applyToTargets(nodeIndices, nodes, (n) => ((n.attributes as any).outputTypeSpec = { types, allowOther }));
  },
});

export const OutputConditionalMapToken = createAttrTokenClass({
  type: "OUTPUT_COND_MAP",
  // 値側に $var と [] / [n] を許容
  regex: new RegExp(
    `^=[A-Za-z0-9_]+:\\$${VAR_ID}(?:\\[\\]|\\[\\d+\\])?(?:\\/[A-Za-z0-9_]+:\\$${VAR_ID}(?:\\[\\]|\\[\\d+\\])?)*`
  ),
  priority: 85,
  description: "出力条件マッピング（単一ペアOK）",
  placement: "after",
  symbol: "=",
  doc: { symbol: "=key:value", description: "条件付き出力マップ。キー→変数名を列挙し、結果を分岐保存。", example: "^判定 =pass:$ok/fail:$retry" },
  apply: ({ token, nodes, nodeIndices }) => {
    const raw = token.value.slice(1);
    const pairs = raw.split("/");
    const map: Record<string, string> = {};
    for (const pair of pairs) {
      const [k, v] = pair.split(":");
      if (k && v) map[k.trim()] = v.trim(); // "$ok" など $ を保持
    }
    applyToTargets(nodeIndices, nodes, (n) => ((n.attributes as any).outputConditionalMap = map));
  },
});

export const LoopTimesToken = createAttrTokenClass({
  type: "LOOP_TIMES",
  regex: /^×[1-3]/,
  priority: 85,
  description: "ループ回数（最大3）",
  placement: "after",
  symbol: "×",
  doc: { description: "同一タスクの繰り返し回数を制限付きで指定 (1-3)。", example: "^アイデア出し ×3" },
  apply: ({ token, nodes, nodeIndices }) => {
    const n = Math.max(1, Math.min(3, parseInt(token.value.replace("×", ""), 10)));
    applyToTargets(nodeIndices, nodes, (node) => ((node.attributes as any).loopTimes = n));
  },
});

export const CheckToken = createAttrTokenClass({
  type: "CHECK",
  regex: /^✓/,
  priority: 10,
  description: "要確認",
  placement: "after",
  symbol: "✓",
  doc: { description: "要確認フラグ。人間レビュー待ちタスクとして扱う。", example: "^レビュー✓" },
  apply: ({ nodes, nodeIndices }) => {
    applyToTargets(nodeIndices, nodes, (n) => (n.attributes["check"] = true));
  },
});

export const OutputOffToken = createAttrTokenClass({
  type: "SEMICOLON",
  regex: /^;/,
  priority: 10,
  description: "出力渡さない",
  placement: "after",
  symbol: ";",
  doc: { description: "後続ノードへ出力を渡さない（副作用のみ）指定。", example: "^UI整備;" },
  apply: ({ nodes, nodeIndices }) => {
    applyToTargets(nodeIndices, nodes, (n) => (n.attributes["outputOff"] = true));
  },
});

export const KnowledgeKeyToken = createAttrTokenClass({
  type: "KNOWLEDGE_KEY",
  regex: /^#[A-Za-z0-9_\-\.]+/,
  priority: 86,
  description: "辞書キー（タスク結合に使用）",
  placement: "after",
  symbol: "#",
  doc: { description: "ナレッジ辞書を参照するキー。Knowledge モジュールが解決。", example: "^tipsのデザイン#designTips" },
  apply: ({ token, nodes, nodeIndices }) => {
    const key = token.value.slice(1);
    applyToTargets(nodeIndices, nodes, (n) => ((n.attributes as any).knowledgeKey = key));
  },
});

/* Operators */
export const PipeToken = createOperatorTokenClass({
  type: "PIPE",
  symbol: "|",
  regex: /^\|/,
  priority: 5,
  description: "順次",
  layout: "row",
  doc: { description: "順次接続オペレーター。左→右に出力を引き継ぐ。", example: "^リサーチ | ^要約" },
});
export const AmpersandToken = createOperatorTokenClass({
  type: "AMPERSAND",
  symbol: "&",
  regex: /^&/,
  priority: 4,
  description: "並列",
  layout: "row",
  doc: { description: "強並列接続。左右ノードは独立実行される。", example: "^A & ^B" },
});
export const InheritanceToken = createOperatorTokenClass({
  type: "INHERITANCE",
  symbol: ">",
  regex: /^>/,
  priority: 3,
  description: "継承",
  layout: "column",
  abbreviation: "F",
  doc: { description: "継承接続。親チェーンの属性を引き継ぎ上書き。", example: "^P > ^C1 > ^C2" },
});
export const WeakParallelToken = createOperatorTokenClass({
  type: "WEAK_PARALLEL",
  symbol: "§",
  regex: /^§/,
  priority: 2,
  description: "弱並列",
  layout: "column",
  abbreviation: "S",
  doc: { description: "弱並列。行並列用。＆は列並列用。", example: "^調査 § ^検証" },
});
export const DownArrowToken = createOperatorTokenClass({
  type: "DOWN_ARROW",
  symbol: "↓",
  // before: regex: /^[↓\n]+/,
  regex: /^↓+/,
  priority: 1,
  description: "弱順次/改行",
  layout: "column",
  abbreviation: "R",
  doc: { description: "レイアウト上の区切り。実行上は弱順次。", example: "^A ↓ ^B" },
});

/* Unknown */
export class UnknownToken extends KNLToken {
  static symbol = "·";
  static regex = /.^/;
  static type = "UNKNOWN";
  static priority = 0;
  static description = "未知";
  static doc = { description: "既知のトークンにマッチしない断片。デバッグ向け。" };
  constructor(v: string, p = 0) {
    super(UnknownToken.type, v, p, UnknownToken.description);
  }
}

/* Registry + First-char dispatch */
export const KNL07_TOKEN_CLASSES: KNL07TokenConstructor[] = [
  PersonaNodeToken,
  UndecidedTaskToken as unknown as KNL07TokenConstructor,
  TaskToken as unknown as KNL07TokenConstructor,
  VariableNodeToken,

  DescriptionToken,
  ExplicitInputToken,
  RequirementInputToken,
  ExplicitOutputToken,
  OutputTypeSpecToken,
  OutputConditionalMapToken,
  LoopTimesToken,
  CheckToken,
  OutputOffToken,
  KnowledgeKeyToken,

  PipeToken,
  AmpersandToken,
  InheritanceToken,
  WeakParallelToken,
  DownArrowToken,
];

export const SORTED_KNL07_TOKEN_CLASSES: KNL07TokenConstructor[] = [...KNL07_TOKEN_CLASSES].sort(
  (a, b) => (b.priority ?? 0) - (a.priority ?? 0),
);

const KNL07TokenRegistry = (() => {
  type Def = {
    ctor: KNL07TokenConstructor;
    firstChars: Set<string>;
    priority: number;
  };
  const defs: Def[] = [];
  const firstCharMap = new Map<string, Def[]>();

  function guessFirstChars(ctor: KNL07TokenConstructor): Set<string> {
    const set = new Set<string>();
    const sym = (ctor as any).symbol as string | undefined;
    if (sym && sym.length) set.add(sym[0]);
    const src = ctor.regex.source;
    const m = src.match(/^\^\[?\\?([^\]\|\\])/); // 先頭の 1 文字（^" や ^\() を拾う簡易ヒューリスティック
    if (m && m[1]) set.add(m[1]);
    if (src.includes("\\n")) set.add("\n");
    return set;
  }

  function register(ctor: KNL07TokenConstructor) {
    const first = guessFirstChars(ctor);
    const def: Def = { ctor, firstChars: first, priority: ctor.priority ?? 0 };
    defs.push(def);
    first.forEach((ch) => {
      const arr = firstCharMap.get(ch) || [];
      arr.push(def);
      firstCharMap.set(ch, arr);
    });
  }

  function finalize() {
    defs.sort((a, b) => b.priority - a.priority);
    firstCharMap.forEach((arr, k) => firstCharMap.set(k, arr.sort((a, b) => b.priority - a.priority)));
  }

  function candidates(ch: string): Def[] {
    if (!KNL07Flags.tokenizerFirstCharDispatch) return defs;
    const arr = firstCharMap.get(ch);
    return (arr && arr.length) ? arr : defs;
  }

  SORTED_KNL07_TOKEN_CLASSES.forEach(register);
  finalize();
  return { candidates };
})();

/* Tokenizer + Incremental */
export function knl07Tokenize(input: string): KNLToken[] {
  KNL07Metrics.incr("tokenize.calls");
  let cur = input;
  const tokens: KNLToken[] = [];
  let pos = 0;
  let idx = 0;

  // 改行は基本無視。space/tabだけを通常スキップ用に定義
  const skipSpaces = () => {
    const m = cur.match(/^[ \t]+/);
    if (m) {
      cur = cur.slice(m[0].length);
      pos += m[0].length;
    }
  };

  while (cur.length > 0) {
    // A) 例外規則: 「改行群 + 任意のspace/tab + 直後が^」→ 暗黙の↓ を1つ注入
    //    （^ が続くときだけ列ブレークとして扱う。それ以外の改行は全部捨てる）
    const nlThenCaret = cur.match(/^(?:\r?\n)+[ \t]*(?=\^)/);
    if (nlThenCaret) {
      // 暗黙の↓を1つ注入
      const T: any = DownArrowToken;
      const t: KNLToken = new T("↓", pos);
      t.tokenIndex = idx++;
      tokens.push(t);

      // 改行＋空白部分だけ消費（^ は残す）
      cur = cur.slice(nlThenCaret[0].length);
      pos += nlThenCaret[0].length;

      // 次ループへ（このあと ^ が通常のTaskToken等としてマッチする）
      continue;
    }

    // B) 改行その他の空白を丸ごと無視（Aに引っかからない改行は捨てる）
    const anyWs = cur.match(/^\s+/);
    if (anyWs) {
      cur = cur.slice(anyWs[0].length);
      pos += anyWs[0].length;
      continue;
    }

    // C) ここから通常のトークンマッチ
    let matched = false;
    const ch = cur[0];
    const defs = KNL07TokenRegistry.candidates(ch);

    for (const def of defs) {
      const C = def.ctor;
      const m = cur.match(C.regex);
      if (m && m[0]) {
        const v = m[0];
        const T: any = C;
        const t: KNLToken = new T(v, pos);
        t.tokenIndex = idx++;
        tokens.push(t);
        cur = cur.slice(v.length);
        pos += v.length;
        matched = true;
        break;
      }
    }

    if (!matched) {
      const m = cur.match(new RegExp(`^${NON_TOKEN_CLASS}`));
      const v = m?.[0] || cur[0];
      const t = new UnknownToken(v, pos);
      t.tokenIndex = idx++;
      tokens.push(t);
      cur = cur.slice(v.length);
      pos += v.length;
    }

    // トークン直後の space/tab は捌く（改行は捨てない。次ループのBで丸ごと無視）
    skipSpaces();
  }

  return tokens;
}

export function knl07IncrementalTokenize(prevDsl: string, prevTokens: KNLToken[], nextDsl: string): KNLToken[] {
  KNL07Metrics.incr("tokenize.incremental.calls");
  let prefix = 0;
  while (prefix < prevDsl.length && prefix < nextDsl.length && prevDsl[prefix] === nextDsl[prefix]) prefix++;
  let suffix = 0;
  while (suffix < (prevDsl.length - prefix) && suffix < (nextDsl.length - prefix) &&
    prevDsl[prevDsl.length - 1 - suffix] === nextDsl[nextDsl.length - 1 - suffix]) suffix++;

  const leftBoundary = prefix;
  const rightBoundaryPrev = prevDsl.length - suffix;

  const preservedLeft = prevTokens.filter(t => (t.position + t.value.length) <= leftBoundary);
  const shift = nextDsl.length - prevDsl.length;
  const preservedRight = prevTokens.filter(t => t.position >= rightBoundaryPrev)
    .map(t => {
      const clone = Object.assign(Object.create(Object.getPrototypeOf(t)), t);
      clone.position += shift;
      return clone;
    });

  const middleNew = nextDsl.slice(leftBoundary, nextDsl.length - suffix);
  const newMiddleTokens = knl07Tokenize(middleNew).map(tok => {
    tok.position += leftBoundary;
    return tok;
  });

  const merged = [...preservedLeft, ...newMiddleTokens, ...preservedRight];
  merged.forEach((t, i) => (t.tokenIndex = i));
  return merged;
}

// [Spec] C-3 Parser
// Purpose: トークン列から NodeSet/ExecutionBlock を組み立てて静的実行グラフと I/O を算出。
// Inputs: `KNLToken[]`, DSL文字列, 属性ポリシー。
// Outputs: `KNL07ParseResult` (nodes, executionTree, rows, nodePositions)。
// Guarantees: 演算子優先度に従い deterministic にブロック化、from/toリンク・計算済み入力を付与。
// Downstream: Knowledge(C-4)、Prompt(C-5)、Runtime(C-7)、UI全般で消費。

/* ================================================================
 * C-3: Syntax Parser
 * ================================================================ */
export interface KNL07NodeSet {
  id: string;
  type: "NODE_SET";
  primaryNode: NodeToken; // Task-like
  attachedNodes: NodeToken[];
  nodeindex?: number;
  value: string;
  connector?: string;
  entries: NodeToken[];
  exits: NodeToken[];
}

export interface KNL07ExecutionBlock {
  id: string;
  type: "BLOCK";
  operator: OperatorToken;
  children: (KNL07NodeSet | KNL07ExecutionBlock)[];
  abbreviation: string;
  connector?: string;
  entries: NodeToken[];
  exits: NodeToken[];
}

export type KNL07Chainable = KNL07NodeSet | KNL07ExecutionBlock;
export type KNL07Item = KNL07Chainable | OperatorToken  ;

export interface KNL07ParseResult {
  nodes: NodeToken[];
  executionTree: KNL07Chainable[];
  rows: KNL07Chainable[][];
  nodePositions: Record<number, { row: number; stepInRow: number }>;
}

export interface KNL07ParseOutput {
  tokens: KNLToken[];
  parsed: KNL07ParseResult;
}

const CONNECTOR_SYMBOLS: Record<string, string> = {
  PIPE: "→",
  AMPERSAND: "&",
  INHERITANCE: "↪",
  WEAK_PARALLEL: "§",
};

const PRECEDENCE_CLASSES: Array<new (...args: any[]) => OperatorToken> = [
  PipeToken as unknown as new (...args: any[]) => OperatorToken,
  AmpersandToken as unknown as new (...args: any[]) => OperatorToken,
  InheritanceToken as unknown as new (...args: any[]) => OperatorToken,
  WeakParallelToken as unknown as new (...args: any[]) => OperatorToken,
];

/* ver5: 小ユーティリティ化 */
function makeConnectorSymbol(op: OperatorToken): string {
  return CONNECTOR_SYMBOLS[op.type as keyof typeof CONNECTOR_SYMBOLS] ?? "";
}
function isChainable(item: KNL07Item): item is KNL07Chainable {
  return Boolean(item) && typeof (item as any).type === "string" &&
    (((item as any).type === "NODE_SET") || ((item as any).type === "BLOCK"));
}
function isConnectiveOperator(op: OperatorToken): boolean {
  return op.type === "PIPE" || op.type === "INHERITANCE";
}

function connectEdges(fromNodes: NodeToken[], toNodes: NodeToken[]): void {
  for (const from of fromNodes) {
    for (const to of toNodes) {
      if (typeof from.nodeindex === "number" && typeof to.nodeindex === "number") {
        if (!from.to.includes(to.nodeindex)) from.to.push(to.nodeindex);
        if (!to.from.includes(from.nodeindex)) to.from.push(from.nodeindex);
      }
    }
  }
}

function collapseByOperator(items: KNL07Item[], OpClass: new (...args: any[]) => OperatorToken): KNL07Item[] {
  const output: KNL07Item[] = [];

  for (let i = 0; i < items.length; i++) {
    const current = items[i];
    const left = output[output.length - 1];
    const right = items[i + 1];

    if (current instanceof OpClass && isChainable(left) && isChainable(right)) {
      output.pop();
      const op = current as OperatorToken;
      const connector = makeConnectorSymbol(op);
      const connective = isConnectiveOperator(op);

      if ((left as any).type === "BLOCK" && (left as KNL07ExecutionBlock).operator.type === op.type) {
        const block = left as KNL07ExecutionBlock;
        if (connective) connectEdges(block.exits, (right as KNL07Chainable).entries);
        block.children.push(right as KNL07Chainable);
        if (connective) block.exits = (right as KNL07Chainable).exits;
        else {
          block.entries = [...block.entries, ...(right as KNL07Chainable).entries];
          block.exits = [...block.exits, ...(right as KNL07Chainable).exits];
        }
        output.push(block);
      } else {
        const block: KNL07ExecutionBlock = {
          id: `block-${op.type}-${output.length}`,
          type: "BLOCK",
          operator: op,
          children: [left as KNL07Chainable, right as KNL07Chainable],
          abbreviation: (op.constructor as any).abbreviation ?? op.type,
          connector,
          entries: connective ? (left as KNL07Chainable).entries
            : [...(left as KNL07Chainable).entries, ...(right as KNL07Chainable).entries],
          exits: connective ? (right as KNL07Chainable).exits
            : [...(left as KNL07Chainable).exits, ...(right as KNL07Chainable).exits],
        };
        if (connective) connectEdges((left as KNL07Chainable).exits, (right as KNL07Chainable).entries);
        output.push(block);
      }
      i++;
    } else {
      output.push(current);
    }
  }

  return output;
}

function annotateDslRanges(tokens: KNLToken[], originalDsl: string): void {
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i];
    if (token instanceof NodeToken) {
      const prev = tokens[i - 1];
      const leftIsOperator = prev instanceof OperatorToken && !(prev instanceof DownArrowToken);
      let j = i + 1;
      while (j < tokens.length && !(tokens[j] instanceof OperatorToken)) j++;
      const start = leftIsOperator ? prev.position : token.position;
      const end = j < tokens.length ? tokens[j].position : originalDsl.length;
      const node = token as NodeToken;
      node.dslStart = start;
      node.dslEnd = end;
      node.dslRaw = originalDsl.slice(start, end);
      node.dsl = node.dslRaw.trim();
    }
  }
}

function cloneNodeToken(token: NodeToken, nodes: NodeToken[], typeCounters: Record<string, number>): NodeToken {
  const node = Object.assign(Object.create(Object.getPrototypeOf(token)), token) as NodeToken;
  node.nodeindex = nodes.length;
  typeCounters[node.type] = (typeCounters[node.type] ?? 0) + 1;
  node.typeIndex = typeCounters[node.type];
  node.guid = makeStableNodeGuid(node.type, node.value, node.position, node.nodeindex ?? nodes.length);
  nodes.push(node);
  return node;
}

function applyAttributes(
  attr: AttrToken,
  nodeList: NodeToken[],
  primary: NodeToken,
  nodes: NodeToken[],
  policy: AttrPolicy,
) {
  const listIdxs = nodeList.map((n) => n.nodeindex).filter((idx): idx is number => typeof idx === "number");
  const primaryIdx = primary.nodeindex ?? 0;
  const lastIdx = listIdxs[listIdxs.length - 1];
  const target = lastIdx !== undefined ? [lastIdx] : [];
  attr.assignAttr(nodes, target, policy, primaryIdx, listIdxs);
}

function segmentByRows(items: KNL07Item[]): KNL07Item[][] {
  const rows: KNL07Item[][] = [[]];
  for (const item of items) {
    if (item instanceof DownArrowToken) rows.push([]);
    else rows[rows.length - 1].push(item);
  }
  return rows.filter((row) => row.length > 0);
}

function collapseRows(rows: KNL07Item[][]): KNL07Chainable[][] {
  return rows.map((row) => {
    let list: KNL07Item[] = row;
    for (const Op of PRECEDENCE_CLASSES) list = collapseByOperator(list, Op);
    return list.filter(isChainable) as KNL07Chainable[];
  });
}

function extractNodeSets(chain: KNL07Chainable): KNL07NodeSet[] {
  return chain.type === "NODE_SET"
    ? [chain as KNL07NodeSet]
    : (chain as KNL07ExecutionBlock).children.flatMap((child) => extractNodeSets(child as KNL07Chainable));
}

function mapNodePositions(rows: KNL07Chainable[][]): Record<number, { row: number; stepInRow: number }> {
  const positions: Record<number, { row: number; stepInRow: number }> = {};
  rows.forEach((row, rowIdx) => {
    const nodeSets = row.flatMap((chain) => extractNodeSets(chain));
    nodeSets.forEach((set, setIdx) => {
      const index = set.primaryNode.nodeindex;
      if (typeof index === "number") positions[index] = { row: rowIdx + 1, stepInRow: setIdx + 1 };
    });
  });
  return positions;
}

function calculateNodeIO(nodes: NodeToken[]): void {
  nodes.forEach((node) => {
    // v4 と同等: TaskToken のみ対象（UndecidedTaskToken は TaskToken 継承で包含）
    if (!(node instanceof (TaskToken as any))) return;
    const inputs = node.explicitInputs.map((input) => ({ fromNode: -1, input }));
    node.from.forEach((fromIdx) => {
      const fromNode = nodes[fromIdx];
      if (!fromNode || fromNode.attributes["outputOff"]) return;
      if (fromNode.explicitOutputs.length) {
        fromNode.explicitOutputs.forEach((output) => {
          inputs.push({ fromNode: fromIdx, input: output });
        });
      } else {
        inputs.push({ fromNode: fromIdx, input: `Result of Node[${fromIdx}]` });
      }
    });
    node.calculatedInputs = inputs;
  });
}

export function parseKNL07Tokens(tokens: KNLToken[], originalDsl: string, policy: AttrPolicy = "lastOnly"): KNL07ParseResult {
  annotateDslRanges(tokens, originalDsl);

  const nodes: NodeToken[] = [];
  const typeCounters: Record<string, number> = {};
  tokens.forEach((token) => {
    if (token instanceof NodeToken) {
      cloneNodeToken(token, nodes, typeCounters);
    }
  });

  const items: KNL07Item[] = [];
  const consumed = new Set<number>();

  for (let i = 0; i < tokens.length; i++) {
    if (consumed.has(i)) continue;
    const token = tokens[i];

    if (token instanceof UnknownToken) {
      consumed.add(i);
      continue;
    }
    if (token instanceof DownArrowToken) {
      items.push(token);
      consumed.add(i);
      continue;
    }
    if (token instanceof OperatorToken) {
      items.push(token);
      consumed.add(i);
      continue;
    }
    if (token instanceof (TaskToken as any)) {
      const primary = nodes.find((node) => node.id === token.id) as NodeToken;
      let j = i + 1;
      while (j < tokens.length && !(tokens[j] instanceof OperatorToken)) j++;

      const nodeSet: KNL07NodeSet = {
        id: `set-${primary.id}`,
        type: "NODE_SET",
        primaryNode: primary,
        attachedNodes: [],
        nodeindex: primary.nodeindex,
        value: primary.value,
        exits: [primary],
        entries: [primary],
      };
      consumed.add(i);
      const nodeList: NodeToken[] = [primary];

      for (let k = i + 1; k < j; k++) {
        const subToken = tokens[k];
        consumed.add(k);
        if (subToken instanceof NodeToken) {
          if (subToken.id !== primary.id) nodeSet.attachedNodes.push(subToken);
          nodeList.push(subToken);
        } else if (subToken instanceof AttrToken) {
          applyAttributes(subToken, nodeList, primary, nodes, policy);
        }
      }

      const personas = nodeSet.attachedNodes
        .filter((node) => node.type === "PERSONA_NODE")
        .map((node) => node.value.replace(/^@/, ""));
      if (personas.length) (primary.attributes as any).persona = personas.join(", ");

      items.push(nodeSet);
      i = j - 1;
    }
  }

  const segments = segmentByRows(items);
  const collapsed = collapseRows(segments);

  for (let idx = 0; idx < collapsed.length - 1; idx++) {
    const exits = collapsed[idx].flatMap((chain) => chain.exits);
    const entries = collapsed[idx + 1].flatMap((chain) => chain.entries);
    connectEdges(exits, entries);
  }

  const executionTree = collapsed.flat();
  const nodePositions = mapNodePositions(collapsed);
  calculateNodeIO(nodes);

  return {
    nodes,
    executionTree,
    rows: collapsed,
    nodePositions,
  };
}

export function parseKNL07(dsl: string, policy: AttrPolicy = "lastOnly"): KNL07ParseResult {
  const tokens = knl07Tokenize(dsl);
  return parseKNL07Tokens(tokens, dsl, policy);
}

export function parseKNL07WithTokens(dsl: string, policy: AttrPolicy = "lastOnly"): KNL07ParseOutput {
  const tokens = knl07Tokenize(dsl);
  const parsed = parseKNL07Tokens(tokens, dsl, policy);
  return { tokens, parsed };
}

// [Spec] C-4 Knowledge
// Purpose: タスク/ペルソナ/変数辞書を解析結果へ統合し、属性・制約・入力/出力ヒントを補完。
// Inputs: 任意の `KNL07Knowledge`、parser結果。
// Outputs: enrichment 済み `KNL07ParseResult`、binding/confidence・personaスタイルなどを付与。
// Consumers: Prompt(C-5), Runtime(C-7), UI詳細ビュー、計画生成が辞書情報を利用。
// Notes: 辞書が空の場合は非破壊でそのまま返す。

/* ================================================================
 * C-4: Knowledge (tasks/personas/vars)
 * ================================================================ */
export type KNL07KnowledgeTaskEntry = {
  name?: string;
  aliases?: string[];
  description?: string;
  focus?: string;
  scope?: string;
  constraints?: string[];
  inputs?: string[];
  outputs?: string[];
  notes?: string;
};

export type KNL07KnowledgePersonaEntry = {
  displayName?: string;
  style?: string;
  personality?: string;
  policy?: string;
  systemPrompt?: string;
  aliases?: string[];
  notes?: string;
};

export type KNL07KnowledgeVarEntry = {
  name?: string;
  type?: string;
  description?: string;
  examples?: string[];
  aliases?: string[];
  notes?: string;
};

export interface KNL07Knowledge {
  tasks?: Record<string, KNL07KnowledgeTaskEntry>;
  personas?: Record<string, KNL07KnowledgePersonaEntry>;
  vars?: Record<string, KNL07KnowledgeVarEntry>;
}

export interface KNL07AnalyzeOptions {
  policy?: AttrPolicy;
  knowledge?: KNL07Knowledge | null | undefined;
}

function normalizeKnowledgeKey(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

function extractTaskBaseName(taskValue: string): string {
  return (taskValue ?? "").replace(/^\^\??/, "").replace(/["@\(=].*$/, "").trim();
}

function resolveTaskKey(node: NodeToken, tasks: NonNullable<KNL07Knowledge["tasks"]>): string | undefined {
  const explicitKey = normalizeKnowledgeKey((node.attributes as any).knowledgeKey);
  if (explicitKey && tasks[explicitKey]) return explicitKey;

  const base = normalizeKnowledgeKey(extractTaskBaseName(node.value));
  if (tasks[base]) return base;

  const match = Object.entries(tasks).find(([_, entry]) => (entry.aliases ?? []).map(normalizeKnowledgeKey).includes(base));
  return match?.[0];
}

function mergeTaskDictionary(node: NodeToken, taskEntry: KNL07KnowledgeTaskEntry): void {
  const attrs = node.attributes as Record<string, any>;
  if (taskEntry.name && !attrs.taskCanonical) attrs.taskCanonical = taskEntry.name;
  if (taskEntry.description && !attrs.description) attrs.description = taskEntry.description;
  if (taskEntry.focus && !attrs.focus) attrs.focus = taskEntry.focus;
  if (taskEntry.scope && !attrs.scope) attrs.scope = taskEntry.scope;
  if (Array.isArray(taskEntry.constraints) && !attrs.constraints) attrs.constraints = [...taskEntry.constraints];
  if (Array.isArray(taskEntry.inputs) && node.explicitInputs.length === 0) node.explicitInputs.push(...taskEntry.inputs);
  if (Array.isArray(taskEntry.outputs) && node.explicitOutputs.length === 0) node.explicitOutputs.push(...taskEntry.outputs);
}

function mergePersonaDictionary(node: NodeToken, personas: NonNullable<KNL07Knowledge["personas"]>): void {
  const attrs = node.attributes as Record<string, any>;
  const personaKey = normalizeKnowledgeKey(String(attrs.persona ?? "").split(",")[0]);
  if (!personaKey) return;

  let personaEntry: KNL07KnowledgePersonaEntry | undefined = personas[personaKey];
  if (!personaEntry) {
    personaEntry = Object.entries(personas).find(([_, entry]) => (entry.aliases ?? []).map(normalizeKnowledgeKey).includes(personaKey))?.[1];
  }
  if (!personaEntry) return;

  if (personaEntry.style && !attrs.personaStyle) attrs.personaStyle = personaEntry.style;
  if (personaEntry.personality && !attrs.personaPersonality) attrs.personaPersonality = personaEntry.personality;
  if (personaEntry.policy && !attrs.personaPolicy) attrs.personaPolicy = personaEntry.policy;
  if (personaEntry.displayName && !attrs.personaDisplayName) attrs.personaDisplayName = personaEntry.displayName;
}

export function applyKnowledgeToParsedNodes(parsed: KNL07ParseResult, knowledge: KNL07Knowledge | null | undefined): KNL07ParseResult {
  if (!knowledge) return parsed;

  const tasks = knowledge.tasks ?? {};
  const personas = knowledge.personas ?? {};

  for (const node of parsed.nodes) {
    if (!((node as any) instanceof (TaskToken as any))) continue;

    const key = resolveTaskKey(node, tasks);
    if (key) {
      const normalizedExplicit = normalizeKnowledgeKey((node.attributes as any).knowledgeKey);
      (node.attributes as any).binding = { key, confidence: key === normalizedExplicit ? 1 : 0.9 };
      mergeTaskDictionary(node, tasks[key]!);
    }
    if ((node.attributes as any).persona) mergePersonaDictionary(node, personas);
  }
  return parsed;
}

export function enrichKNL07(parsed: KNL07ParseResult, knowledge: KNL07Knowledge | null | undefined): KNL07ParseResult {
  return applyKnowledgeToParsedNodes(parsed, knowledge);
}

export function analyzeKNL07(dsl: string, options: KNL07AnalyzeOptions = {}): KNL07ParseResult {
  const { policy = "lastOnly", knowledge } = options;
  const parsed = parseKNL07(dsl, policy);
  return knowledge ? enrichKNL07(parsed, knowledge) : parsed;
}

// [Spec] C-5 Prompt Builder
// Purpose: ノード属性・辞書情報・実行位置から LLM に渡す最終プロンプトを構築。
// Inputs: NodeToken, 全ノード、コンテキスト、継承インデックス、グローバルWish/Backbone。
// Outputs: personaシステム文や出力制約を含む文字列プロンプト、継承インデックス生成ユーティリティ。
// Consumers: Runtime(C-7) の実行フェーズ、UIのプロンプト確認タブ。
// Notes: 変数出力タグ仕様（`$name{"value":...}`）をここで強制。

/* ================================================================
 * C-5: Prompt Builder
 * ================================================================ */
export const KNL07PromptTemplates = {
  baseSystemPrompt: "あなたは KNL DSL の実行マネージャです。DSL から得られるノード情報をもとに、タスクごとの入出力と要件遵守を支援してください。",
  planGenerator: "未定タスク (^?) に対して、目的・制約・必要な前提知識を短く整理し、3 ステップ以内のプラン候補を提示してください。",
  nodeExecution: "タスク {taskLabel} を実行します。入力: {inputs}。期待する出力タイプ: {outputType}。要件: {requirements}。",
  requirementCheck: "以下の要件が満たされているか検証し、不足があれば再実行指示を出してください: {requirementList}。",
};

export type KNL07InheritanceIndex = Record<number, number[]>;

export function buildKNL07InheritanceIndex(executionTree: KNL07Chainable[]): KNL07InheritanceIndex {
  const map: KNL07InheritanceIndex = {};
  const extractSets = (chain: KNL07Chainable): KNL07NodeSet[] =>
    chain.type === "NODE_SET" ? [chain as KNL07NodeSet] : (chain as KNL07ExecutionBlock).children.flatMap((child) => extractSets(child as KNL07Chainable));
  const visit = (chain: KNL07Chainable) => {
    if (chain.type !== "BLOCK") return;
    const block = chain as KNL07ExecutionBlock;
    if (block.operator.type === "INHERITANCE") {
      for (let i = 0; i < block.children.length - 1; i++) {
        const leftSets = extractSets(block.children[i] as KNL07Chainable);
        const rightSets = extractSets(block.children[i + 1] as KNL07Chainable);
        for (const left of leftSets) {
          for (const right of rightSets) {
            const childIdx = right.primaryNode.nodeindex ?? -1;
            const parentIdx = left.primaryNode.nodeindex ?? -1;
            if (childIdx < 0 || parentIdx < 0) continue;
            if (!map[childIdx]) map[childIdx] = [];
            if (!map[childIdx].includes(parentIdx)) map[childIdx].unshift(parentIdx);
          }
        }
      }
    }
    block.children.forEach((child) => visit(child as KNL07Chainable));
  };
  executionTree.forEach((chain) => visit(chain));
  return map;
}

export function buildKNL07PersonaInstruction(attrs: Record<string, any>): string | undefined {
  const parts: string[] = [];
  if (attrs.personaDisplayName || attrs.persona) parts.push(`あなたは${attrs.personaDisplayName || attrs.persona}です。`);
  if (attrs.personaStyle) parts.push(`文体・スタイル: ${attrs.personaStyle}`);
  if (attrs.personaPersonality) parts.push(`性格: ${attrs.personaPersonality}`);
  if (attrs.personaPolicy) parts.push(`ポリシー: ${attrs.personaPolicy}`);
  return parts.length ? parts.join("\n") : undefined;
}

export function sanitizeKNL07VarKey(value: string): string {
  return String(value ?? "").replace(/^=\$?/, "").replace(/^\$+/, "").replace(/\[\]|\[\d+\]$/, "");
}

function mergeInheritedAttributes(node: NodeToken, nodes: NodeToken[], index: KNL07InheritanceIndex): Record<string, any> {
  const collected: NodeToken[] = [];
  const seen = new Set<number>();
  let frontier = index[node.nodeindex ?? -1] || [];
  while (frontier.length) {
    const next: number[] = [];
    for (const parentIdx of frontier) {
      if (seen.has(parentIdx)) continue;
      seen.add(parentIdx);
      const parent = nodes.find((n) => n.nodeindex === parentIdx);
      if (parent) {
        collected.push(parent);
        const parentsParents = index[parentIdx] || [];
        next.push(...parentsParents);
      }
    }
    frontier = next;
  }
  const merged: Record<string, any> = {};
  const mergeOne = (source: Record<string, any>) => {
    Object.entries(source || {}).forEach(([key, value]) => {
      if (value == null) return;
      if (Array.isArray(value)) merged[key] = Array.from(new Set([...(merged[key] ?? []), ...value]));
      else if (typeof value === "object") merged[key] = { ...(merged[key] ?? {}), ...value };
      else if (merged[key] == null) merged[key] = value;
    });
  };
  collected.reverse().forEach((parent) => mergeOne(parent.attributes as any));
  mergeOne(node.attributes as any);
  return merged;
}

function getExpectedVarNames(node: NodeToken): string[] {
  const outputs = Array.isArray((node as any).explicitOutputs) ? ((node as any).explicitOutputs as string[]) : [];
  const names = outputs.map((value) => sanitizeKNL07VarKey(value).trim()).filter(Boolean);
  return Array.from(new Set(names));
}

export interface KNL07PromptParams {
  node: NodeToken;
  allNodes: NodeToken[];
  ctx: Record<string, any>;
  inheritance?: KNL07InheritanceIndex;
  global?: string;
  backbone?: string;
  position?: { row: number; stepInRow: number };
}


export function buildKNL07FinalPrompt({
  node,
  allNodes,
  ctx,
  inheritance,
  global,
  backbone,
  position,
}: KNL07PromptParams): string {
  const attrs = node.attributes as Record<string, any>;
  const merged = inheritance ? mergeInheritedAttributes(node, allNodes, inheritance) : attrs;
  const flatCtx: Record<string, any> = {
    ...(ctx || {}),
    ...((ctx as any)?.vars || {}),
    ...((ctx as any)?.tags || {}),
  };

  const binding = attrs?.binding as { key?: string; confidence?: number } | undefined;

  let prompt = "";
  if (global?.trim()) prompt += `## Global Wish/Goal\n${global.trim()}\n\n`;
  if (backbone?.trim()) prompt += `## Backbone\n${backbone.trim()}\n\n`;

  const upstream = node.from.map((idx) => allNodes.find((n) => n.nodeindex === idx)?.value || `Node[${idx}]`).join(", ");
  const downstream = node.to.map((idx) => allNodes.find((n) => n.nodeindex === idx)?.value || `Node[${idx}]`).join(", ");

  prompt += "## Positioning\n";
  if (position) prompt += `- Row: R${position.row}, Step: ${position.stepInRow}\n`;
  if (inheritance && (inheritance[node.nodeindex ?? -1]?.length || 0) > 0) {
    const parents = (inheritance[node.nodeindex ?? -1] || []).map((idx) => allNodes.find((n) => n.nodeindex === idx)?.value || `Node[${idx}]`);
    prompt += `- Inherits from: ${parents.join(" -> ")}\n`;
  }
  prompt += upstream ? `- Upstream: ${upstream}\n` : `- Start node\n`;
  prompt += downstream ? `- Downstream: ${downstream}\n` : `- Terminal\n`;

  prompt += `\n## Task\n- Name: ${node.value}${binding?.key ? `  (dict:${binding.key})` : ""}\n`;
  if (binding?.confidence != null) prompt += `- Binding confidence: ${binding.confidence}\n`;
  if ((merged as any).taskCanonical) prompt += `- Canonical: ${(merged as any).taskCanonical}\n`;
  if ((merged as any).description) prompt += `- Description: ${(merged as any).description}\n`;
  if ((merged as any).focus) prompt += `- Focus: ${(merged as any).focus}\n`;
  if ((merged as any).scope) prompt += `- Scope: ${(merged as any).scope}\n`;
  if (Array.isArray((merged as any).constraints) && (merged as any).constraints.length) {
    prompt += `- Constraints:\n${(merged as any).constraints.map((c: string) => `  - ${c}`).join("\n")}\n`;
  }

  const personaSystem = buildKNL07PersonaInstruction({ ...merged, ...attrs });
  if (personaSystem) prompt += `\n## Persona (System hints)\n${personaSystem}\n`;

  // ... flatCtx 定義までは現状のまま ...

// Context Inputs（fromNode 由来は $var 名を優先, 無ければ node_{from}_result へフォールバック）
const inputs: string[] = [];

const serialize = (v: any) => {
  if (v && typeof v === "object" && "summary" in v) return String(v.summary);
  try { return JSON.stringify(v); } catch { return String(v); }
};

node.calculatedInputs.forEach((ci) => {
  const isUpstream = ci.fromNode !== -1;
  const raw = ci.input || ""; // 例: "$ok" / "Result of Node[3]" / "$pages[]"
  const implicitKey = isUpstream ? `node_${ci.fromNode}_result` : undefined;

  // 候補キー: 1) $var 2) var（$を外す） 3) node_{from}_result（フォールバック）
  const keyCandidates: string[] = [];
  if (raw) {
    if (raw.startsWith("$")) {
      // $pages[] / $pages[0] なども sanitize せずまずはそのまま試す
      keyCandidates.push(raw, raw.slice(1));
      // サニタイズ名（[]/[n]を外した素の名前）も一応見る
      const base = sanitizeKNL07VarKey(raw);
      if (base && base !== raw.replace(/^\$/, "")) keyCandidates.push(`$${base}`, base);
    } else {
      keyCandidates.push(raw, `$${raw}`);
    }
  }
  if (implicitKey) keyCandidates.push(implicitKey);

  const found = keyCandidates.find((k) => k in flatCtx);
  if (!found) return;

  // 表示ラベルは、$var があればその“変数名”で出す。なければ node_{from}_result。
  const label =
    raw.startsWith("$")
      ? KNL07Core.sanitizeVarKey(raw) || raw.replace(/^\$/, "")
      : (isUpstream && implicitKey ? implicitKey : raw);

  inputs.push(`- ${label}: ${serialize(flatCtx[found])}`);
});

if (inputs.length) prompt += `\n## Context Inputs\n${inputs.join("\n")}\n`;
  const typeSpec = (attrs as any).outputTypeSpec as { types: string[]; allowOther: boolean } | undefined;
  if (typeSpec) {
    prompt += `\n## Output Type Rule\n- Choose outType: ${typeSpec.types.join(", ")}${typeSpec.allowOther ? ", ." : ""}\n`;
  }
  const condMap = (attrs as any).outputConditionalMap as Record<string, string> | undefined;
  if (condMap && Object.keys(condMap).length) {
    const pairs = Object.entries(condMap).map(([key, value]) => `  - ${key} -> ${value}`).join("\n");
    prompt += `- Mapping hints:\n${pairs}\n`;
  }

  const expectedVars = getExpectedVarNames(node);
  prompt += "\n## Variable Outputs\n";
  if (expectedVars.length) {
    prompt += `- 末尾で下記の各変数を1行につき1つずつ必ず出力してください（順不同）:\n${expectedVars.map((name) => `  - $${name}`).join("\n")}\n`;
  } else {
    prompt += "- 必要に応じて有用な変数を出力してください（任意）。\n";
  }
  prompt += '- 形式: $name{"desc":"短い説明","value":...}（波括弧内は厳密なJSON。ダブルクオート/末尾カンマ禁止）\n';
  prompt += "- これらの行は---詳細---セクションの末尾に追加してください。\n";
  prompt += "\n## Instruction\n- Produce the deliverable considering the above.\n- Output format:\n---要旨---\n{要約}\n---詳細---\n{詳細}\n\n";
  return prompt;
}

// [Spec] C-6 Plans & Coverage
// Purpose: ^? ノードからLLMプラン生成し、要件適合状況とDSL差し替えを管理。
// Inputs: NodeToken, parse結果, 現在のctxとノード状態、LLM呼び出し関数、グローバルwish/backbone。
// Outputs: 代替DSLスニペット、要件フィードバック構造、既存DSLへの採用処理。
// Consumers: ランタイムの未決タスクハンドリング、UIのプラン差し替えUX、履歴タブ。
// Notes: JSONレスポンスのバリデーションと再解析 (parse→enrich) をここで完結。

/* ================================================================
 * C-6: Plans (^? generator) + Requirement Coverage + DSL adoption
 * ================================================================ */
export type KNL07PlanMessage = { role: "user" | "assistant"; content: string };
export type KNL07ConversationMessage = { role: "user" | "assistant"; content: string };

export interface KNL07RequirementCoverage {
  requirement: string;
  inPrompt: boolean;
  inHistory: boolean;
  feedback?: string;
  status?: "pass" | "fail";
}

export interface KNL07RequirementReport {
  requirements: string[];
  coverage: KNL07RequirementCoverage[];
  missing: string[];
  satisfied: boolean;
}

export type KNL07UndecidedPlanFeedback = {
  target: string;
  status: "pass" | "fail";
  message: string;
};

export interface KNL07UndecidedPlanResult {
  planId: string;
  dsl: string;
  feedback: KNL07UndecidedPlanFeedback[];
  notes?: string;
  raw: string;
  prompt: string;
  parsed: KNL07ParseResult;
}

export interface KNL07PlanNodeStateSnapshot {
  status?: string;
  result?: { summary?: string; details?: string } | null;
}

export interface KNL07PlanLLMCallArgs {
  history: KNL07PlanMessage[];
  systemPrompt: string;
  engine: "gemini" | "claude";
  apiKey: string;
}

export type KNL07PlanLLMCaller = (args: KNL07PlanLLMCallArgs) => Promise<string>;

function extractFirstJsonObject<T = any>(text: string): T | null {
  const codeMatch = text.match(/```json([\s\S]*?)```/i);
  const candidateSource = codeMatch ? codeMatch[1] : text;
  const start = candidateSource.indexOf("{");
  const end = candidateSource.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) return null;
  const candidate = candidateSource.slice(start, end + 1);
  try {
    return JSON.parse(candidate) as T;
  } catch {
    return null;
  }
}

export function normalizeKNL07PlanFeedback(raw: any): KNL07UndecidedPlanFeedback[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => {
      const target = String(item?.target ?? "").trim();
      const statusRaw = String(item?.status ?? "").toLowerCase();
      const status: "pass" | "fail" = statusRaw === "fail" ? "fail" : "pass";
      const message = String(item?.message ?? "").trim();
      if (!target) return null;
      return { target, status, message } as KNL07UndecidedPlanFeedback;
    })
    .filter((item): item is KNL07UndecidedPlanFeedback => !!item);
}

export function describeKNL07NodeForPlan(node: NodeToken, state?: KNL07PlanNodeStateSnapshot): string {
  const label = extractTaskBaseName(node.value) || node.value;
  const summary = state?.result?.summary;
  const status = state?.status && state.status !== "completed" ? ` (${state.status})` : "";
  if (summary) return `[${node.nodeindex}] ${label}${status} => ${safeTruncate(summary, 120)}`;
  return `[${node.nodeindex}] ${label}${status}`;
}

/* ver5: 除外ルールをフラグで切替え可能に */
export function summarizeKNL07ContextKeys(
  ctx: Record<string, any>,
  limit = KNL07Flags.limitContextSnapshot,
): string {
  const keys = Object.keys(ctx || {}).filter((key) => {
    if (!key) return false;
    if (KNL07Flags.summarizeExcludeNodeKeys && key.startsWith("node_")) return false;
    if (KNL07Flags.summarizeExcludeResultKeys && key.endsWith("_result")) return false;
    return true;
  });
  if (!keys.length) return "なし";
  const picked = keys.slice(0, limit).map((key) => {
    const value = safeTruncate(ctx[key], 80).replace(/\s+/g, " ").trim();
    return `${key}: ${value}`;
  });
  const suffix = keys.length > limit ? ` …(+${keys.length - limit})` : "";
  return picked.join(", ") + suffix;
}

export interface KNL07GeneratePlanOptions {
  node: NodeToken;
  parsed: KNL07ParseResult;
  ctx: Record<string, any>;
  engine: "gemini" | "claude";
  apiKey: string;
  globalWish: string;
  backbone: string;
  knowledge?: KNL07Knowledge | null;
  nodeStatesSnapshot: Record<number, KNL07PlanNodeStateSnapshot>;
  callLLM: KNL07PlanLLMCaller;
}

export async function generateKNL07PlanForUndecided({
  node,
  parsed,
  ctx,
  engine,
  apiKey,
  globalWish,
  backbone,
  knowledge,
  nodeStatesSnapshot,
  callLLM,
}: KNL07GeneratePlanOptions): Promise<KNL07UndecidedPlanResult> {
  const goal = (node.attributes as any)?.description || extractTaskBaseName(node.value) || node.value;
  const upstream = (node.from || [])
    .map((idx) => parsed.nodes[idx])
    .filter((n): n is NodeToken => !!n)
    .map((n) => describeKNL07NodeForPlan(n, nodeStatesSnapshot[n.nodeindex ?? -999]));
  const downstream = (node.to || [])
    .map((idx) => parsed.nodes[idx])
    .filter((n): n is NodeToken => !!n)
    .map((n) => describeKNL07NodeForPlan(n, nodeStatesSnapshot[n.nodeindex ?? -999]));
  const requirements = (node as any).explicitRequirements as string[] | undefined;
  const varPreview = summarizeKNL07ContextKeys(ctx);
  const placeholderDsl = (node as NodeToken).dsl || node.value;

  const requirementSection = (requirements || []).map((req, i) => `  ${i + 1}. ${req}`).join("\n");

  const planningSystemPrompt = "You are an expert KNL workflow planner. Respond only with JSON matching the requested schema.";

  const promptLines = [
    "# Task",
    `- Goal: ${goal}`,
    "- Produce 1-4 KNL tasks that bridge the upstream context to the downstream expectation.",
    "- Use '^TaskName' syntax with operators like '|' for sequencing.",
    "- Prefer concise yet descriptive task names.",
    "- Assume the snippet will replace the placeholder shown below.",
    "",
    "## Placeholder DSL Segment",
    placeholderDsl,
    "",
    "## Upstream Context",
    upstream.length ? upstream.join("\n") : "  (none)",
    "",
    "## Downstream Expectations",
    downstream.length ? downstream.join("\n") : "  (none)",
    "",
    "## Global Signals",
    `- Global wish: ${globalWish || "(none)"}`,
    `- Backbone: ${backbone || "(none)"}`,
    "",
    "## Available Variables Snapshot",
    `- ${varPreview}`,
  ];
  if (requirements && requirements.length) {
    promptLines.push("", "## (!) Requirements", requirementSection || "  (none)");
  }
  promptLines.push(
    "",
    "## Output JSON Schema",
    "{",
    '  "dsl": "string (valid KNL DSL snippet, no surrounding Markdown)",',
    '  "feedback": [',
    '    {"target": "goal-or-requirement", "status": "pass|fail", "message": "short feedback"}',
    "  ],",
    '  "notes": "(optional short summary)"',
    "}",
    "",
    "Return JSON only. Include one feedback entry for the overall goal and one per requirement (if any).",
  );

  const prompt = promptLines.join("\n");
  const planHistory: KNL07PlanMessage[] = [{ role: "user", content: prompt }];
  const raw = await callLLM({
    history: planHistory,
    systemPrompt: planningSystemPrompt,
    engine,
    apiKey,
  });

  const parsedJson = extractFirstJsonObject<any>(raw);
  if (!parsedJson) throw new Error("LLMプラン応答のJSON解析に失敗しました");
  const dsl = String(parsedJson.dsl || "").trim();
  if (!dsl) throw new Error("生成DSLが空でした");
  const feedback = normalizeKNL07PlanFeedback(parsedJson.feedback);
  const planId = `${node.guid || node.id}-plan-${Date.now().toString(36)}`;

  const planParsed = parseKNL07(dsl, "lastOnly");
  const enriched = knowledge ? applyKnowledgeToParsedNodes(planParsed, knowledge ?? null) : planParsed;

  return {
    planId,
    dsl,
    feedback,
    notes: parsedJson.notes ? String(parsedJson.notes) : undefined,
    raw,
    prompt,
    parsed: enriched,
  };
}

export function evaluateKNL07RequirementCoverage(
  reqs: string[] | undefined,
  prompt: string,
  conversation: KNL07ConversationMessage[],
): KNL07RequirementReport | null {
  const requirements = (reqs || []).map((r) => r.trim()).filter(Boolean);
  if (!requirements.length) return null;
  const normalize = (value: string) => value.toLowerCase();
  const promptText = normalize(prompt);
  const historyText = normalize(conversation.map((m) => m.content).join("\n"));
  const coverage = requirements.map((requirement) => {
    const needle = normalize(requirement);
    const inPrompt = promptText.includes(needle);
    const inHistory = historyText.includes(needle);
    return { requirement, inPrompt, inHistory } as KNL07RequirementCoverage;
  });
  const missing = coverage.filter((item) => !(item.inPrompt && item.inHistory)).map((item) => item.requirement);
  return {
    requirements,
    coverage,
    missing,
    satisfied: missing.length === 0,
  };
}

export function createKNL07RequirementReportFromFeedback(
  requirements: string[] | undefined,
  feedback: KNL07UndecidedPlanFeedback[],
): KNL07RequirementReport | null {
  const list = (requirements || []).map((r) => r.trim()).filter(Boolean);
  if (!list.length) return null;
  const feedbackMap = new Map<string, KNL07UndecidedPlanFeedback>();
  feedback.forEach((item) => {
    if (!feedbackMap.has(item.target)) feedbackMap.set(item.target, item);
  });
  const coverage: KNL07RequirementCoverage[] = list.map((requirement) => {
    const fb = feedbackMap.get(requirement);
    const status = fb?.status === "fail" ? "fail" : "pass";
    return {
      requirement,
      inPrompt: status === "pass",
      inHistory: status === "pass",
      feedback: fb?.message,
      status,
    };
  });
  const missing = coverage.filter((item) => item.status !== "pass").map((item) => item.requirement);
  return {
    requirements: list,
    coverage,
    missing,
    satisfied: missing.length === 0,
  };
}

/* 生成プランの採用: ターゲットノードのDSL範囲を置換（v4踏襲） */
export function adoptKNL07PlanIntoDsl(originalDsl: string, targetNode: NodeToken, planDsl: string): string {
  const start = targetNode.dslStart ?? 0;
  const end = targetNode.dslEnd ?? start;
  const left = originalDsl.slice(0, start);
  const right = originalDsl.slice(end);
  const insert = planDsl.trim();
  const needsNLLeft = left && !left.endsWith("\n") ? "\n" : "";
  const needsNLRight = right && !right.startsWith("\n") ? "\n" : "";
  return `${left}${needsNLLeft}${insert}${needsNLRight}${right}`.replace(/\n{3,}/g, "\n\n");
}

// [Spec] C-7 Execution Runtime & LLM Clients
// Purpose: トポロジ順実行と確認ゲート処理、LLM入出力の共通ハンドリングを提供。
// Inputs: NodeToken配列、実行関数、コンテキスト/履歴、LLMAPIキー、フラグ定義。
// Outputs: 実行状態マップ、LLMレスポンス解析結果、リトライ制御付きAPI呼び出し。
// Consumers: UI Executeタブ、外部統合向けFacade(`KNL07Core.ExecutionRuntime`ほか)。
// Notes: pending_confirmationゲートやリトライポリシー(Flags)はここで一本化。

/* ================================================================
 * C-7: Execution Runtime + Variable / LLM parsing + LLM clients
 * ================================================================ */
export type KNL07NodeStatus =
  | "pending"
  | "ready"
  | "running"
  | "completed"
  | "failed"
  | "pending_confirmation";

export const KNL07_LLM_HISTORY_LIMIT = KNL07Flags.limitHistoryTail;
export const KNL07_CONTEXT_SNAPSHOT_LIMIT = KNL07Flags.limitContextSnapshot;

export function cloneKNL07History(history: KNL07ConversationMessage[]): KNL07ConversationMessage[] {
  return history.map((msg) => ({ ...msg }));
}

export function buildKNL07ContextSnapshotMessages(
  ctx: Record<string, any>,
  limit = KNL07_CONTEXT_SNAPSHOT_LIMIT,
): KNL07ConversationMessage[] {
  if (!ctx) return [];
  const entries = Object.entries(ctx).filter(([key]) => {
    if (!key) return false;
    if (key.startsWith("$")) return true;
    if (key.endsWith("_result")) return true;
    return false;
  });
  if (!entries.length) return [];
  const picked = entries.slice(0, limit);
  const lines = picked.map(([key, value]) => `- ${key}: ${safeTruncate(value, 240)}`);
  if (entries.length > limit) lines.push(`- … (+${entries.length - limit} more entries)`);
  return [{ role: "assistant", content: `# Context Snapshot\n${lines.join("\n")}` }];
}

export function buildKNL07GeneratedTaskPrefixMessages(options: {
  parentHistory: KNL07ConversationMessage[];
  planDsl?: string;
  ctx: Record<string, any>;
}): KNL07ConversationMessage[] {
  const { parentHistory, planDsl, ctx } = options;
  const contextMessages = buildKNL07ContextSnapshotMessages(ctx);
  const planMessages: KNL07ConversationMessage[] = planDsl
    ? [{ role: "assistant", content: `# Plan Context\n${safeTruncate(planDsl, 1200)}` }]
    : [];
  const capacity = Math.max(0, KNL07_LLM_HISTORY_LIMIT - contextMessages.length - planMessages.length);
  const parentTail = capacity > 0 ? parentHistory.slice(-capacity) : [];
  return [...parentTail, ...planMessages, ...contextMessages];
}

export interface KNL07NodeState {
  status: KNL07NodeStatus;
  result: { summary: string; details: string } | null;
  history: KNL07ConversationMessage[];
  error: string | null;
  completedAt: string | null;
  requirementReport?: KNL07RequirementReport | null;
}

export function createKNL07BaseNodeState(overrides: Partial<KNL07NodeState> = {}): KNL07NodeState {
  return {
    status: "pending",
    result: null,
    history: [],
    error: null,
    completedAt: null,
    requirementReport: null,
    ...overrides,
  };
}

export function createKNL07InitialNodeStates(tasks: NodeToken[]): Record<number, KNL07NodeState> {
  return tasks.reduce<Record<number, KNL07NodeState>>((acc, task) => {
    if (task.nodeindex != null) acc[task.nodeindex] = createKNL07BaseNodeState();
    return acc;
  }, {});
}

export function patchKNL07NodeState(
  prev: Record<number, KNL07NodeState>,
  nodeIndex: number,
  patch: Partial<KNL07NodeState>,
): Record<number, KNL07NodeState> {
  const current = prev[nodeIndex] ?? createKNL07BaseNodeState();
  return { ...prev, [nodeIndex]: { ...current, ...patch } };
}

/* 変数抽出: $name{...json...} */
export function parseKNL07VariableTags(text: string): Record<string, any> {
  const vars: Record<string, any> = {};
  const regex = /\$([a-zA-Z0-9_]+)\{([\s\S]*?)\}/g;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    const name = m[1];
    const body = m[2].trim();
    try {
      const obj = JSON.parse(body);
      vars[name] = obj;
      continue;
    } catch {}
    try {
      const obj2 = JSON.parse(`{${body}}`);
      vars[name] = obj2;
    } catch {
      // ignore invalid fragments
    }
  }
  return vars;
}

export interface KNL07LLMParsed {
  summary: string;
  details: string;
  variables: Record<string, any>;
}

export function parseKNL07LLM(text: string): KNL07LLMParsed {
  const summaryMatch = text.match(/---要旨---\s*([\s\S]*?)\s*---詳細---/);
  const detailsMatch = text.match(/---詳細---\s*([\s\S]*?)$/);
  const summary = (summaryMatch?.[1] || "").trim();
  const details = (detailsMatch?.[1] || "").trim() || text;
  const variables = parseKNL07VariableTags(text);
  return { summary, details, variables };
}

/* 実行ランタイム（依存解決 + 確認ゲート対応） */
type KNL07TaskNodeIndex = number;

export class KNL07ExecutionRuntime {
  private nodeMap = new Map<KNL07TaskNodeIndex, NodeToken>(); // TaskToken 互換
  private dependencyMap = new Map<KNL07TaskNodeIndex, KNL07TaskNodeIndex[]>();
  private memo = new Map<KNL07TaskNodeIndex, Promise<void>>();
  private executor: ((node: NodeToken) => Promise<void>) | null = null;

  constructor(tasks: NodeToken[]) {
    const taskIndexes = new Set<KNL07TaskNodeIndex>();
    tasks.forEach((task) => {
      if (task.nodeindex == null) return;
      taskIndexes.add(task.nodeindex);
    });
    tasks.forEach((task) => {
      if (task.nodeindex == null) return;
      this.nodeMap.set(task.nodeindex, task);
      const deps = (task.from || []).filter((idx) => taskIndexes.has(idx));
      this.dependencyMap.set(task.nodeindex, deps);
    });
  }

  async runAll(executor: (node: NodeToken) => Promise<void>) {
    this.executor = executor;
    const promises: Promise<void>[] = [];
    this.nodeMap.forEach((_, idx) => {
      promises.push(this.runNode(idx));
    });
    await Promise.all(promises);
  }

  /* UIサイドで確認完了後に targetIndex の状態を更新してから再実行する */
  registerDynamicTask(task: NodeToken, dependencies: KNL07TaskNodeIndex[]) {
    if (task.nodeindex == null) return;
    this.nodeMap.set(task.nodeindex, task);
    this.dependencyMap.set(task.nodeindex, dependencies);
    this.memo.delete(task.nodeindex);
  }

  private runNode(idx: KNL07TaskNodeIndex): Promise<void> {
    if (this.memo.has(idx)) return this.memo.get(idx)!;
    const deps = this.dependencyMap.get(idx) || [];
    const exec = this.executor;
    const promise = Promise.all(deps.map((dep) => this.runNode(dep))).then(() => {
      if (!exec) return;
      const node = this.nodeMap.get(idx);
      if (!node) return;
      return exec(node);
    });
    this.memo.set(idx, promise);
    return promise;
  }
}

/* LLM 共通: retry + timeout（ver5: 共通ラッパ） */
async function callWithRetry<T>(
  fn: (signal: AbortSignal) => Promise<T>,
  label: string,
  {
    maxRetries = KNL07Flags.llmMaxRetries,
    timeoutMs = KNL07Flags.llmTimeoutMs,
    initialBackoffMs = KNL07Flags.llmInitialBackoffMs,
    jitter = KNL07Flags.llmJitter,
  } = {},
): Promise<T> {
  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
  let attempt = 0;
  const run = async (): Promise<T> => {
    attempt++;
    const ctl = new AbortController();
    const timer = setTimeout(() => ctl.abort(), timeoutMs);
    try {
      const r = await fn(ctl.signal);
      clearTimeout(timer);
      return r;
    } catch (e: any) {
      clearTimeout(timer);
      if (!KNL07Flags.llmRetry || attempt > maxRetries) {
        throw new Error(`${label} failed after ${attempt} attempts: ${e?.message || e}`);
      }
      const backoff = initialBackoffMs * Math.pow(2, attempt - 1) * (jitter ? 0.7 + Math.random() * 0.6 : 1);
      await sleep(backoff);
      return run();
    }
  };
  return run();
}

/* Gemini */
export interface KNL07GeminiCallArgs {
  history: KNL07ConversationMessage[];
  apiKey: string;
  model?: string;
  systemInstruction?: string;
  fetchImpl?: typeof fetch;
  endpoint?: string;
}

export async function callKNL07GeminiAPI({
  history,
  apiKey,
  model = "gemini-2.5-flash-preview-05-20",
  systemInstruction,
  fetchImpl,
  endpoint,
}: KNL07GeminiCallArgs): Promise<string> {
  return callWithRetry<string>(async (signal) => {
    if (!apiKey) throw new Error("Gemini API Key未設定");
    const fetcher = fetchImpl ?? globalThis.fetch;
    if (!fetcher) throw new Error("fetch が利用できません");
    const url =
      endpoint ||
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
    const contents = history.map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));
    const body: any = { contents };
    if (systemInstruction) body.systemInstruction = { parts: [{ text: systemInstruction }] };
    const res = await fetcher(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal,
    });
    if (!res.ok) throw new Error(`Gemini Error: ${res.status} ${res.statusText} - ${await res.text()}`);
    const data = await res.json();
    const text = data?.candidates?.[0]?.content?.parts?.[0]?.text || "";
    if (!text) throw new Error("Gemini 応答が空");
    return text;
  }, "Gemini");
}

/* Claude */
export interface KNL07ClaudeCallArgs {
  history: KNL07ConversationMessage[] | string;
  apiKey: string;
  model?: string;
  systemPrompt?: string;
  fetchImpl?: typeof fetch;
}

export async function callKNL07ClaudeAPI({
  history,
  apiKey,
  model = "claude-3.5-sonnet-20241022",
  systemPrompt = "You are a helpful assistant.",
  fetchImpl,
}: KNL07ClaudeCallArgs): Promise<string> {
  return callWithRetry<string>(async (signal) => {
    if (!apiKey) throw new Error("Claude API Key未設定");
    const fetcher = fetchImpl ?? globalThis.fetch;
    if (!fetcher) throw new Error("fetch が利用できません");
    const messages = Array.isArray(history)
      ? history.map((entry) => ({ role: entry.role, content: [{ type: "text", text: entry.content }] }))
      : [{ role: "user", content: [{ type: "text", text: history }] }];
    const res = await fetcher("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-api-key": apiKey,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model,
        max_tokens: 4000,
        system: systemPrompt,
        messages,
      }),
      signal,
    });
    if (!res.ok) throw new Error(`Claude Error: ${res.status} ${res.statusText} - ${await res.text()}`);
    const data = await res.json();
    const text: string = data?.content?.[0]?.text || "";
    if (!text) throw new Error("Claude 応答が空");
    return text;
  }, "Claude");
}

/* 実行ゲート（✓）補助（v4踏襲） */
export function shouldKNL07ExecuteNow(task: NodeToken, state: KNL07NodeState): boolean {
  const needsCheck = !!(task.attributes as any)?.check;
  if (needsCheck && (state.status === "pending" || state.status === "ready")) {
    state.status = "pending_confirmation";
    return false;
  }
  return true;
}

// [Spec] Facade (Public API Bundle)
// Purpose: DSL解析〜実行までの主要ユーティリティを `KNL07Core` に集約し外部へ提供。
// Inputs: 外部呼び出し側が渡すDSL文字列、NodeToken、知識データ、LLMハンドラ。
// Outputs: tokenize/parse結果+enrich、プロンプト生成器、ランタイム/LLMコールヘルパー等。
// Consumers: 他モジュールやUI、APIサーバー（`KNL07Core` をimportして利用）。
// Notes: 内部実装を差し替えても公開契約を守れば互換維持。Flags/Metricsもここからアクセス。

/* ================================================================
 * Facade (public API)
 * ================================================================ */
export const KNL07Core = {
  // Tokenizer/Parser
  tokenize: knl07Tokenize,
  incrementalTokenize: knl07IncrementalTokenize,
  parse(dsl: string, policy: AttrPolicy) {
    const tokens = knl07Tokenize(dsl);
    const parsed = parseKNL07Tokens(tokens, dsl, policy);
    return { tokens, parsed };
  },
  enrich(parsed: KNL07ParseResult, knowledge: KNL07Knowledge | null | undefined) {
    return enrichKNL07(parsed, knowledge ?? null);
  },

  // Prompt/Inheritance
  buildPrompt: buildKNL07FinalPrompt,
  buildInheritanceIndex: buildKNL07InheritanceIndex,
  buildPersonaInstruction: buildKNL07PersonaInstruction,

  // Plans / Requirements
  generatePlanForUndecided: generateKNL07PlanForUndecided,
  evaluateRequirementCoverage: evaluateKNL07RequirementCoverage,
  createRequirementReportFromFeedback: createKNL07RequirementReportFromFeedback,
  adoptPlanIntoDsl: adoptKNL07PlanIntoDsl,

  // LLM parse / Variables
  parseLLM: parseKNL07LLM,
  parseVariableTags: parseKNL07VariableTags,
  sanitizeVarKey: sanitizeKNL07VarKey,

  // Runtime helpers
  ExecutionRuntime: KNL07ExecutionRuntime,
  createInitialNodeStates: createKNL07InitialNodeStates,
  patchNodeState: patchKNL07NodeState,
  buildContextSnapshotMessages: buildKNL07ContextSnapshotMessages,
  buildGeneratedTaskPrefixMessages: buildKNL07GeneratedTaskPrefixMessages,
  cloneHistory: cloneKNL07History,
  shouldExecuteNow: shouldKNL07ExecuteNow,

  // LLM clients
  callGeminiAPI: callKNL07GeminiAPI,
  callClaudeAPI: callKNL07ClaudeAPI,

  // Flags/Metrics
  Flags: KNL07Flags,
  Metrics: KNL07Metrics,
};

// [Spec] U-1/U-2 UI Skeleton
// Purpose: 単一ファイルUIでDSL編集・可視化・実行・知識管理を統合し、核心ロジックを直接呼び出す。
// Inputs: `State`/`Action` リデューサー、`KNL07Core` API、localStorage設定、LLM応答、DSL文字列。
// Outputs: タブ別ビュー(Render)、LLM設定モーダル、DSL Assist、履歴スナップショット、ユーザー操作イベント。
// Consumers: Reactアプリのルート (`KNL07` コンポーネント)、外部埋め込み（将来的に）。
// Notes: MainTab=visualize/swimlane/execute/knowledge/history の遷移粒度、Undo/RedoスタックやDSLスナップショットがグローバルに共有。

// [Map] UI Section Overview
// U-1 HeaderCommon: DSL入力・Toolbar・Tokenizer・Assistモーダル。State/dispatchの中心入口。
// U-2 Visualization: ExecutionBlockViewとWorkflowSvgで構造閲覧、ダブルクリック→NodeDetailへ。
// U-3 NodeDetail & Dict: NodeDetailCard、TaskDictFieldsがDSL編集/辞書同期とプロンプト確認を担当。
// U-4 Knowledge/History Tabs: (後続)テーブル系ビューがState.knowledge/historyを読む。
// U-5 Modals & Settings: LLMSettingsModal, DslAssistModal を共有的に利用。
// U-6 Reducer & Actions: State/Action型とreducerがUI全体のイベントマップを維持。

const UIStyles = {
  twoColumnForm: { display: "grid", gridTemplateColumns: "140px 1fr", rowGap: 8, columnGap: 8 } as React.CSSProperties,
  chipFilterRow: { display: "flex", gap: 6, flexWrap: "wrap" } as React.CSSProperties,
};

/* ================================================================
 * U-1/U-2 UI Skeleton (frame first, content to be migrated gradually)
 * ================================================================ */



/**
 * KNL UI Full Single File (v7.1)
 * - Header: DSL入力/Tokenizer/Inspector + 歯車（LLM設定共有: knl.engine, knl.apiKey, knl.claudeKey）
 * - Visualization: Tree + Workflow(SVG 左→右) + DAG一覧 + グローバルポート/SQLライクレジャー
 * - Knowledge Tab: task/var/persona/tool（thinktool） 辞書参照/LLM新規登録/定義編集/履歴
 * - Execute Tab: DAGトポ順で実行。^?の動的追従（DSL変化時に計画マージ）。✓ゲート（承認待ち/承認/承認メモ/リトライ/差戻し）
 * - History Tab: runId+taskId主キーで履歴共有（実行タブと同ストア）。全文/ラン/タスク絞り込み/詳細JSON表示
 * - Node Editor: ノード詳細とDSL編集、辞書編集、プロンプト確認。@複数（persona/tool）対応
 * - Tool辞書: type/in/work/out/motif/aliases/notes。@でペルソナと同居可。実行時にSystem補助文へ（今は履歴metaへ保存）
 */

/* =============== Small UI primitives =============== */
// [Spec] U-1a Card primitive
// Purpose: シンプルな枠レイアウト。Header/Rightスロットを提供しUI各所で再利用。
// Inputs: title/right(children)、任意style。
// Outputs: Styled container wrapping children。
// Notes: Sticky headerなどでも使うため backgroundやflex指定は呼び出し側が上書き可能。
const Card: React.FC<{
  title?: React.ReactNode;
  right?: React.ReactNode;
  children?: React.ReactNode;
  style?: React.CSSProperties;
}> = ({ title, right, children, style }) => (
  <div className="card" style={style}>
    {(title || right) && (
      <div className="card-header" style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div>{title}</div>
        <div>{right}</div>
      </div>
    )}
    <div className="card-content">{children}</div>
  </div>
);

const GEAR_ICON = (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path d="M12 8a4 4 0 100 8 4 4 0 000-8zm8.94 4a6.94 6.94 0 01-.19 1.6l2.11 1.65-2 3.46-2.49-1a6.98 6.98 0 01-2.77 1.6l-.38 2.65h-4l-.38-2.65A6.98 6.98 0 015.43 18l-2.49 1-2-3.46L3.05 14A6.94 6.94 0 012.86 12c0-.54.06-1.07.19-1.6L.94 8.75l2-3.46 2.49 1A6.98 6.98 0 018.2 4.7L8.58 2h4l.38 2.65a6.98 6.98 0 012.77 1.6l2.49-1 2 3.46-2.11 1.65c.13.53.19 1.06.19 1.6z" fill="#6b7280"/>
  </svg>
);

/* =============== Types / State =============== */
type MainTab = "visualize" | "swimlanebuilder" | "execute" | "knowledge" | "history";
type GlobalHistory = { dslSnapshots: { dsl: string; timestamp: string; note?: string }[] };

type ToolDef = {
  type?: string;
  in?: string[];
  work?: string;
  out?: string[];
  motif?: string;
  aliases?: string[];
  notes?: string;
};
type AnyKnowledge = KNL07Knowledge & { tools?: Record<string, ToolDef> };

type State = {
  dsl: string;
  attrPolicy: AttrPolicy;
  mainTab: MainTab;
  selectedNodeIdx: number | null;
  hoveredNodeIdx: number | null;
  dslUndoStack: string[];
  dslRedoStack: string[];
  globalWish: string;
  backbone: string;
  knowledge: AnyKnowledge | null;
  history: GlobalHistory;
  settingsOpen: boolean;
};

type Action =
  | { type: "SET_DSL"; dsl: string; snapshot?: boolean; note?: string }
  | { type: "UNDO_DSL" }
  | { type: "REDO_DSL" }
  | { type: "SET_ATTR_POLICY"; policy: AttrPolicy }
  | { type: "SET_MAIN_TAB"; tab: MainTab }
  | { type: "SELECT_NODE"; idx: number | null }
  | { type: "HOVER_NODE"; idx: number | null }
  | { type: "SET_GLOBAL"; wish: string }
  | { type: "SET_BACKBONE"; backbone: string }
  | { type: "APPLY_KNOWLEDGE"; knowledge: AnyKnowledge | null }
  | { type: "HIST_ADD_DSL_SNAPSHOT"; dsl: string; note?: string }
  | { type: "OPEN_SETTINGS"; open: boolean };

const nowJP = () => new Date().toLocaleString("ja-JP");
const setDslAction = (dsl: string, opt: { snapshot?: boolean; note?: string } = {}): Action => ({
  type: "SET_DSL",
  dsl,
  snapshot: opt.snapshot,
  note: opt.note,
});
const undoDslAction = (): Action => ({ type: "UNDO_DSL" });
const redoDslAction = (): Action => ({ type: "REDO_DSL" });
const setAttrPolicyAction = (policy: AttrPolicy): Action => ({ type: "SET_ATTR_POLICY", policy });
const setMainTabAction = (tab: MainTab): Action => ({ type: "SET_MAIN_TAB", tab });
const selectNodeAction = (idx: number | null): Action => ({ type: "SELECT_NODE", idx });
const setGlobalWishAction = (wish: string): Action => ({ type: "SET_GLOBAL", wish });
const setBackboneAction = (backbone: string): Action => ({ type: "SET_BACKBONE", backbone });
const applyKnowledgeAction = (knowledge: AnyKnowledge | null): Action => ({ type: "APPLY_KNOWLEDGE", knowledge });
const addDslSnapshotAction = (dsl: string, note?: string): Action => ({ type: "HIST_ADD_DSL_SNAPSHOT", dsl, note });
const openSettingsAction = (open: boolean): Action => ({ type: "OPEN_SETTINGS", open });

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "SET_DSL": {
      if (action.dsl === state.dsl) return state;
      const MAX = 200;
      const undo = [...state.dslUndoStack, state.dsl].slice(-MAX);
      const next: State = { ...state, dsl: action.dsl, dslUndoStack: undo, dslRedoStack: [] };
      if (action.snapshot) {
        next.history = {
          ...state.history,
          dslSnapshots: [
            ...state.history.dslSnapshots,
            { dsl: action.dsl, timestamp: nowJP(), note: action.note },
          ],
        };
      }
      return next;
    }
    case "UNDO_DSL": {
      if (!state.dslUndoStack.length) return state;
      const MAX = 200;
      const prev = state.dslUndoStack.at(-1)!;
      const undo = state.dslUndoStack.slice(0, -1);
      const redo = [...state.dslRedoStack, state.dsl].slice(-MAX);
      return { ...state, dsl: prev, dslUndoStack: undo, dslRedoStack: redo };
    }
    case "REDO_DSL": {
      if (!state.dslRedoStack.length) return state;
      const MAX = 200;
      const nextDsl = state.dslRedoStack.at(-1)!;
      const redo = state.dslRedoStack.slice(0, -1);
      const undo = [...state.dslUndoStack, state.dsl].slice(-MAX);
      return { ...state, dsl: nextDsl, dslUndoStack: undo, dslRedoStack: redo };
    }
    case "SET_ATTR_POLICY":
      return { ...state, attrPolicy: action.policy };
    case "SET_MAIN_TAB":
      return { ...state, mainTab: action.tab };
    case "SELECT_NODE":
      return { ...state, selectedNodeIdx: action.idx };
    case "HOVER_NODE":
      return { ...state, hoveredNodeIdx: action.idx };
    case "SET_GLOBAL":
      return { ...state, globalWish: action.wish };
    case "SET_BACKBONE":
      return { ...state, backbone: action.backbone };
    case "APPLY_KNOWLEDGE":
      return { ...state, knowledge: action.knowledge };
    case "HIST_ADD_DSL_SNAPSHOT":
      return {
        ...state,
        history: {
          ...state.history,
          dslSnapshots: [
            ...state.history.dslSnapshots,
            { dsl: action.dsl, timestamp: nowJP(), note: action.note },
          ],
        },
      };
    case "OPEN_SETTINGS":
      return { ...state, settingsOpen: action.open };
    default:
      return state;
  }
}

/* =============== 共通 LLM 設定（歯車） =============== */
// [Spec] U-5a LLMSettingsModal
// Purpose: LLMエンジンとAPIキーをlocalStorage共有スコープへ保存。
// Inputs: open/onClose。内部でengine/apiKey stateを持ち、open時にlocalStorage読み出し。
// Outputs: 保存ボタンでlocalStorage更新し閉じる。
// Notes: HeaderCommonのギアボタンやその他から呼ばれる想定。
const LLMSettingsModal: React.FC<{
  open: boolean;
  onClose: () => void;
}> = ({ open, onClose }) => {
  const [engine, setEngine] = React.useState<"gemini" | "claude">((localStorage.getItem("knl.engine") as any) || "gemini");
  const [apiKey, setApiKey] = React.useState<string>("");

  React.useEffect(() => {
    if (!open) return;
    const keyStorage = engine === "gemini" ? "knl.apiKey" : "knl.claudeKey";
    setApiKey(localStorage.getItem(keyStorage) || "");
  }, [open, engine]);

  if (!open) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h4 className="m-0">LLM 設定（共通）</h4>
          <button onClick={onClose} className="btn-icon">×</button>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "160px 1fr", gap: 8 }}>
          <label>エンジン</label>
          <select className="select" value={engine} onChange={(e) => setEngine(e.target.value as any)}>
            <option value="gemini">Gemini</option>
            <option value="claude">Claude</option>
          </select>
          <label>API Key</label>
          <input className="input" type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder={engine === "gemini" ? "Gemini API Key" : "Claude API Key"} />
        </div>
        <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
          <button
            className="btn btn-primary"
            onClick={() => {
              localStorage.setItem("knl.engine", engine);
              if (engine === "gemini") localStorage.setItem("knl.apiKey", apiKey);
              else localStorage.setItem("knl.claudeKey", apiKey);
              onClose();
            }}
          >
            保存
          </button>
          <button className="btn" onClick={onClose}>閉じる</button>
        </div>
      </div>
    </div>
  );
};

/* =============== DSL Assist（共通設定を使用） =============== */
// [Spec] U-5b DslAssistModal
// Purpose: DSLの案生成をLLM経由で行い、結果を挿入/置換/追記するUI。
// Inputs: open/onClose、現在コンテキスト(wish/backbone/dsl)、挿入/置換/追記ハンドラ。
// Outputs: ideas配列(最大3)をボタン操作で親へ送る。
// Notes: 共通LLM設定(localStorage)を利用。エラー表示や生成状態を内部で管理。
const DslAssistModal: React.FC<{
  open: boolean;
  onClose: () => void;
  context: { wish: string; backbone: string; currentDsl: string };
  onInsertAtCursor: (dsl: string) => void;
  onReplaceDsl: (dsl: string) => void;
  onAppendDsl: (dsl: string) => void;
}> = ({ open, onClose, context, onInsertAtCursor, onReplaceDsl, onAppendDsl }) => {
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [ideas, setIdeas] = React.useState<string[]>([]);

  const generate = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    setIdeas([]);
    try {
      // Strict DSL generation rules (output contract):
      const sys = [
        "あなたは KNL DSL の提案アシスタントです。必ず仕様に完全準拠した DSL だけを生成します。",
        "",
        "【出力形式 重要】",
        "- 出力は <DSL> と </DSL> に厳密に挟まれた本文のみ。説明・箇条書き・コードブロック・Markdownは禁止。",
        "- 1回の応答で最大3案まで。各案はそれぞれ独立した <DSL>...</DSL> ブロックで返す。",
        "",
        "【許可される要素・記号】",
        "- ノード: ^name または ^?name（nameは[a-zA-Z0-9_\-]+）",
        "- 演算子: |（行内順次）, &（行内並列）, >（継承）, §（行並列）, ↓（行順次）",
        "- 修飾: @persona（複数可）, =$var, ($in1)($in2)…, =!type/. , =k:v, ;（文区切り）, ✓, ×n, #key, \"desc\"",
        "- 変数名: $[a-zA-Z0-9_\-]+（$は1回のみ）",
        "",
        "【結合・優先順位（左から強→弱）】",
        "1) |（行内順次）、2) &（行内並列）、3) >（継承）、4) §（行並列）、5) ↓（行順次）",
        "- 明示的な括弧は使用しない。上記優先と左結合に従う。",
        "",
        "【スタイル規約】",
        "- 1つの論理行には1つ以上のノード。; で文を区切る。無駄な空白は避ける。",
        "- ノード名・変数名は簡潔。日本語は原則 \"desc\" のみで使用。",
        "- 不明点は ^? を使い、後段で確定できるよう最小限の仮置きをする。",
        "",
        "【短い例】",
        "<DSL>",
        "^Collect =$topic @research | ^Draft ($topic)=$draft @writer ↓ ^Review ($draft)=$rev ✓",
        "</DSL>",
        "",
        "仕様に反する文字・Markdown・説明文を含めないこと。常に <DSL>…</DSL> のみを返すこと。",
      ].join("\n");
      const prompt = [
        `Goal: ${context.wish || "(none)"}`,
        `Backbone: ${context.backbone || "(none)"}`,
        "Current DSL:",
        context.currentDsl || "(empty)",
        "---",
        "最大3案の <DSL>…</DSL> を返してください。",
      ].join("\n");
      const history = [{ role: "user" as const, content: prompt }];
      let text = "";
      const engine = (localStorage.getItem("knl.engine") as any) || "gemini";
      if (engine === "gemini") {
        const apiKey = localStorage.getItem("knl.apiKey") || "";
        text = await KNL07Core.callGeminiAPI({ history, apiKey, systemInstruction: sys });
      } else {
        const apiKey = localStorage.getItem("knl.claudeKey") || "";
        text = await KNL07Core.callClaudeAPI({ history, apiKey, systemPrompt: sys });
      }
      // Extract strictly the <DSL> blocks only
      const rawBlocks = Array.from(text.matchAll(/<DSL>[\s\S]*?<\/DSL>/g)).map((m) => m[0]);
      const cleaned = rawBlocks
        .map((blk) => blk.replace(/^<DSL>/, "").replace(/<\/DSL>$/, "").trim())
        .map((s) => s.replace(/```[\s\S]*?```/g, "").trim()); // guard: remove any accidental code fences

      // Basic validator: ensure no markdown headings and only allowed chars
      const allowed = /^[\s\S]*$/; // liberal for now; grammar is enforced by downstream tokenizer
      const valid = cleaned.filter((s) => s && !/^\s*[#*\-`]/m.test(s) && allowed.test(s));

      if (!valid.length) {
        throw new Error("<DSL>…</DSL> 形式の候補が見つかりませんでした。プロンプトルールに従い、説明文やMarkdownを含めないでください。");
      }
      // de-duplicate and cap to 3
      const uniq = Array.from(new Set(valid)).slice(0, 3);
      setIdeas(uniq);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }, [context]);

  if (!open) return null;
  const engineLabel = (localStorage.getItem("knl.engine") as any) || "gemini";
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h4 className="m-0">DSL Assist <span style={{ fontSize: 12, color: "#6b7280" }}>(Engine: {engineLabel})</span></h4>
          <div className="flex items-center gap-2">
            <button className="btn btn-primary" onClick={generate} disabled={loading}>
              {loading ? "生成中…" : "生成"}
            </button>
            <button className="btn" onClick={onClose}>閉じる</button>
          </div>
        </div>
        {error && <div className="mb-2 text-sm text-rose-700">{error}</div>}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          {ideas.map((idea, idx) => (
            <Card key={idx} title={<div className="font-semibold">案 {idx + 1}</div>}>
              <pre className="whitespace-pre-wrap text-sm">{idea}</pre>
              <div className="mt-2 flex gap-2 flex-wrap">
                <button className="btn" onClick={() => onInsertAtCursor(idea)}>カーソルに挿入</button>
                <button className="btn" onClick={() => onAppendDsl(idea)}>末尾に追記</button>
                <button className="btn btn-primary" onClick={() => onReplaceDsl(idea)}>置き換え</button>
              </div>
            </Card>
          ))}
        </div>
        {!ideas.length && !loading && (
          <div className="text-sm text-gray-600">「生成」を押すと3案を提案します。</div>
        )}
      </div>
    </div>
  );
};

/* =============== Header（Inspector統合） =============== */
// [Spec] U-1 HeaderCommon
// Purpose: DSL編集の中心ハブ。Global/Backbone設定、Tokenizer表示、Assistモーダル起動、Undo/Redo管理。
// Inputs: state/dispatch、tokens、parsed、onOpenNode。
// Outputs: DSL textarea更新・スナップショット・モーダル制御・シンボル挿入イベント。
// Notes: Stickyヘッダーでツールバー常駐。State更新はすべてアクション経由でReducerと同期。
const SYMBOL_BUTTONS = ["^", "^?", "$", "@", "|", "&", ">", "§", "↓", "($)", "(!)", "=$", "=!", '""', ";", "✓", "×2", "×3", "#"];

const HeaderCommon: React.FC<{
  state: State;
  dispatch: React.Dispatch<Action>;
  tokens: KNLToken[];
  parsed: KNL07ParseResult | null;
  onOpenNode: (idx: number) => void;
}> = ({ state, dispatch, tokens, parsed, onOpenNode }) => {
  const taRef = React.useRef<HTMLTextAreaElement>(null);
  const [assistOpen, setAssistOpen] = React.useState(false);

  const insert = (s: string) => {
    const ta = taRef.current;
    if (!ta) return;
    const { selectionStart, selectionEnd, value } = ta;
    const txt = value.slice(0, selectionStart) + s + value.slice(selectionEnd);
    dispatch(setDslAction(txt, { snapshot: true, note: "toolbar" }));
    setTimeout(() => {
      ta.focus();
      ta.selectionStart = ta.selectionEnd = selectionStart + s.length;
    }, 0);
  };

  return (
    <Card
      title={<h3 className="m-0">DSL Header</h3>}
      right={
        <div className="flex items-center gap-2">
          <button className="btn" title="LLM設定（共通）" onClick={() => dispatch(openSettingsAction(true))}>
            {GEAR_ICON}
          </button>
          <label className="text-xs">Attr適用:</label>
          <select
            className="select max-w-[180px]"
            value={state.attrPolicy}
            onChange={(e) => dispatch(setAttrPolicyAction(e.target.value as AttrPolicy))}
          >
            <option value="lastOnly">最後のノード</option>
            <option value="primaryOnly">primaryのみ</option>
            <option value="setAll">セット全体</option>
          </select>
        </div>
      }
      style={{ position: "sticky", top: 0, zIndex: 20, background: "#f7f8fa" }}
    >
      {/* Prompt-to-DSL-to-Graph helper */}
      <div className="text-xs" style={{ margin: "6px 0 8px", color: "#6b7280" }}>
        自然文を書いて「AIでDSL生成」を押すと、AIがDSLを組み、可視化タブで即グラフが見られます。
      </div>
      {/* Global / Backbone */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div>
          <label className="text-xs">Global Wish/Goal</label>
          <textarea
            className="textarea"
            value={state.globalWish}
            onChange={(e) => dispatch(setGlobalWishAction(e.target.value))}
          />
        </div>
        <div>
          <label className="text-xs">Backbone</label>
          <textarea
            className="textarea"
            value={state.backbone}
            onChange={(e) => dispatch(setBackboneAction(e.target.value))}
          />
        </div>
      </div>

      {/* DSL Toolbar */}
      <div className="mt-2 toolbar" style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
        {/* AI primary CTA */}
        <button className="btn btn-primary" onClick={() => setAssistOpen(true)} title="AIでDSLを自動生成">
          AIでDSL生成
        </button>
        {SYMBOL_BUTTONS.map((s) => (
          <button key={s} onClick={() => insert(s)} className="chip-button">
            {s}
          </button>
        ))}
        <button onClick={() => dispatch(addDslSnapshotAction(state.dsl, "snapshot"))} className="btn">
          Snapshot
        </button>
        {/* 既存Assistは上のプライマリCTAに統合 */}
        <button
          className="chip-button"
          onClick={() => dispatch(undoDslAction())}
          disabled={!state.dslUndoStack.length}
        >
          Undo
        </button>
        <button
          className="chip-button"
          onClick={() => dispatch(redoDslAction())}
          disabled={!state.dslRedoStack.length}
        >
          Redo
        </button>
      </div>

      {/* DSL Input + Tokenizer */}
      <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12 }}>
        <div>
          <strong>DSL Input</strong>
          <textarea
            className="textarea text-[15px] min-h-[120px]"
            ref={taRef}
            value={state.dsl}
            onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
              const key = e.key.toLowerCase();
              if (e.ctrlKey && !e.shiftKey && key === "z") {
                e.preventDefault();
                dispatch(undoDslAction());
              } else if ((e.ctrlKey && key === "y") || (e.ctrlKey && e.shiftKey && key === "z")) {
                e.preventDefault();
                dispatch(redoDslAction());
              }
            }}
            onChange={(e) => dispatch(setDslAction(e.target.value))}
          />
        </div>
        <div>
          <strong>Tokenizer Result</strong>
          <div className="bg-white border border-slate-200 rounded-lg p-2 max-h-40 overflow-y-auto font-mono text-xs">
            {tokens.map((t: KNLToken, i: number) => (
              <div key={t.id} className="border-b border-slate-100 py-0.5">
                <span className="text-slate-400">[{i}]</span> <b>{t.type}</b>: "{t.value}"{" "}
                <span className="text-sky-600">pos:{t.position}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Node Inspector（共通） */}
      <Card title={<h4 className="m-0">Node Inspector（共通）</h4>} style={{ marginTop: 12 }}>
        <div style={{ display: "flex", flexWrap: "wrap" }}>
          {parsed?.nodes.map((n) => (
            <button
              key={n.id}
              onClick={() => n.nodeindex != null && onOpenNode(n.nodeindex)}
              className="chip-button"
              style={{ marginRight: 8, marginBottom: 8 }}
            >
              <b>[{n.nodeindex}]</b> {n.value}
            </button>
          ))}
          {!parsed?.nodes.length && <span style={{ color: "#6b7280" }}>ノードがありません</span>}
        </div>
      </Card>

      <DslAssistModal
        open={assistOpen}
        onClose={() => setAssistOpen(false)}
        context={{ wish: state.globalWish, backbone: state.backbone, currentDsl: state.dsl }}
        onInsertAtCursor={(dsl) => {
          const ta = taRef.current;
          const start = ta?.selectionStart ?? state.dsl.length;
          const end = ta?.selectionEnd ?? start;
          const next = state.dsl.slice(0, start) + dsl + state.dsl.slice(end);
          dispatch(setDslAction(next, { snapshot: true, note: "DSL Assist: insert" }));
          setTimeout(() => {
            if (ta) {
              ta.focus();
              ta.selectionStart = ta.selectionEnd = start + dsl.length;
            }
          }, 0);
        }}
        onAppendDsl={(dsl) =>
          dispatch(
            setDslAction(
              state.dsl + (state.dsl && !state.dsl.endsWith("\n") ? "\n" : "") + dsl,
              { snapshot: true, note: "DSL Assist: append" },
            ),
          )
        }
        onReplaceDsl={(dsl) => dispatch(setDslAction(dsl, { snapshot: true, note: "DSL Assist: replace" }))}
      />
    </Card>
  );
};

/* =============== ExecutionBlockView (Tree) =============== */
// [Spec] U-2a ExecutionBlockView
// Purpose: DSL構文木(ExecutionBlock/NodeSet)を再帰的に描画。ダブルクリックでノード編集を開く。
// Inputs: item(ExecutionBlock/NodeSet)、onNodeDoubleClick。
// Outputs: 再帰ツリーDOM、attachedNodesやconnector描画。
// Notes: BLOCKとNODE_SETを切替え、layout(row/column)とconnector記号を描画。
const ExecutionBlockView: React.FC<{
  item: any;
  onNodeDoubleClick?: (nodeIdx: number | null) => void;
}> = ({ item, onNodeDoubleClick }) => {
  if (item?.type === "BLOCK") {
    const isRow = item.operator.layout === "row";
    const header = (
      <div
        style={{
          textAlign: "center",
          fontWeight: "bold",
          color: isRow ? "#2b3ac0" : "#c0392b",
          marginBottom: 8,
          borderBottom: "1px dashed #ddd",
          paddingBottom: 4,
        }}
      >
        {item.operator.type}
      </div>
    );
    const wrapChild = (child: any, idx: number) => (
      <div key={child?.id ?? idx} style={{ display: "flex", alignItems: "center" }}>
        <ExecutionBlockView item={child} onNodeDoubleClick={onNodeDoubleClick} />
        {idx < item.children.length - 1 && item.connector && (
          <span style={{ margin: "0 6px", color: "#888", fontWeight: 700 }}>{item.connector}</span>
        )}
      </div>
    );
    return (
      <div style={{ border: "1px solid #ccc", borderRadius: 8, padding: 8, background: "rgba(0,0,0,0.02)" }}>
        {header}
        <div style={{ display: "flex", flexDirection: isRow ? "row" : "column", gap: 8 }}>
          {item.children.map(wrapChild)}
        </div>
      </div>
    );
  }

  const set = item ?? {};
  const primary = set.primaryNode as NodeToken | undefined;
  const isTerminated = primary?.attributes?.["outputOff"];

  return (
    <div
      onDoubleClick={() => onNodeDoubleClick?.(set.nodeindex ?? null)}
      style={{
        background: "#fff",
        border: "1px solid #ddd",
        borderRadius: 6,
        padding: 8,
        opacity: isTerminated ? 0.6 : 1,
        cursor: "pointer",
      }}
    >
      <div>
        <b style={{ color: "#2563eb" }}>Node[{set.nodeindex ?? "-"}]</b>: {primary?.value || "-"}
      </div>
      <div style={{ fontSize: 12, marginTop: 2 }}>
        <span style={{ color: "#0a7" }}>in:</span> {primary?.explicitInputs?.join(", ") || "-"}
        <span style={{ color: "#b00", marginLeft: 12 }}>out:</span> {primary?.explicitOutputs?.join(", ") || "-"}
      </div>
      {!!set.attachedNodes?.length && (
        <div style={{ paddingLeft: 16, marginTop: 4, borderLeft: "2px solid #eee" }}>
          {set.attachedNodes.map((an: NodeToken, idx: number) => (
            <div key={an.id ?? idx} style={{ fontSize: 12, padding: "2px 0", color: "#333" }}>
              <span
                style={{
                  background: "#e0e0e0",
                  padding: "2px 6px",
                  borderRadius: 4,
                  marginRight: 8,
                  display: "inline-block",
                  minWidth: 60,
                  textAlign: "center",
                }}
              >
                {an.type.split("_")[0]}
              </span>
              <span>{an.value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* =============== Visualization（Tree/Workflow） =============== */
// [Spec] U-2b WorkflowSvg
// Purpose: トポロジー順のレイアウトを算出しSVGでDAGを描画。ノードクリックで詳細オープン。
// Inputs: parsed (KNL07ParseResult)、onOpenNode。
// Outputs: SVGノード/エッジ。layoutヘルパー(createL2RLayout)で座標管理。
// Notes: Persona表示や終端透明度など軽微な可視化アクセントを付与。
type L2RLayout = {
  width: number;
  height: number;
  nodeW: number;
  nodeH: number;
  coord: Map<number, { x: number; y: number }>;
};

function createL2RLayout(
  nodes: NodeToken[],
  colGap = 260,
  rowGap = 110,
  nodeW = 220,
  nodeH = 64,
  margin = 40,
): L2RLayout {
  const map = new Map<number, NodeToken>();
  nodes.forEach((n) => {
    if (n.nodeindex != null) map.set(n.nodeindex, n);
  });
  // Build adjacency
  const indeg = new Map<number, number>();
  const out = new Map<number, number[]>();
  const inn = new Map<number, number[]>();
  map.forEach((_, idx) => {
    indeg.set(idx, 0);
    out.set(idx, []);
    inn.set(idx, []);
  });
  map.forEach((n, idx) => {
    n.to.forEach((t) => {
      if (!map.has(t)) return;
      out.get(idx)!.push(t);
      inn.get(t)!.push(idx);
      indeg.set(t, (indeg.get(t) || 0) + 1);
    });
  });

  // Topological order (Kahn)
  const q: number[] = [];
  indeg.forEach((d, i) => d === 0 && q.push(i));
  const topo: number[] = [];
  while (q.length) {
    const u = q.shift()!;
    topo.push(u);
    (out.get(u) || []).forEach((v) => {
      const d = (indeg.get(v) || 0) - 1;
      indeg.set(v, d);
      if (d === 0) q.push(v);
    });
  }
  // Fallback for cycles (shouldn't happen but be safe)
  map.forEach((_, idx) => {
    if (!topo.includes(idx)) topo.push(idx);
  });

  // Longest-path layering (sources = 0)
  const layerOf = new Map<number, number>();
  topo.forEach((i) => {
    const preds = inn.get(i)!;
    const base = preds.length ? Math.max(...preds.map((p) => (layerOf.get(p) || 0) + 1)) : 0;
    layerOf.set(i, base);
  });

  // Group into layers
  const layersArr: number[][] = [];
  layerOf.forEach((d, i) => {
    if (!layersArr[d]) layersArr[d] = [];
    layersArr[d].push(i);
  });
  // Initial order by node index
  layersArr.forEach((arr) => arr.sort((a, b) => a - b));

  // Barycenter ordering sweeps (reduce crossings)
  const position = new Map<number, number>();
  const rebuildPositions = () => {
    layersArr.forEach((layer, li) => layer.forEach((v, k) => position.set(v, k)));
  };
  rebuildPositions();

  const barySort = (targets: number[], neighbors: (v: number) => number[]) => {
    const scored = targets.map((v) => {
      const ns = neighbors(v).filter((n) => position.has(n));
      const bc = ns.length ? ns.reduce((s, n) => s + (position.get(n) || 0), 0) / ns.length : position.get(v)!;
      return { v, bc };
    });
    scored.sort((a, b) => a.bc - b.bc || a.v - b.v);
    return scored.map((s) => s.v);
  };

  for (let sweep = 0; sweep < 3; sweep++) {
    // top-down
    for (let li = 1; li < layersArr.length; li++) {
      layersArr[li] = barySort(layersArr[li], (v) => inn.get(v) || []);
      rebuildPositions();
    }
    // bottom-up
    for (let li = layersArr.length - 2; li >= 0; li--) {
      layersArr[li] = barySort(layersArr[li], (v) => out.get(v) || []);
      rebuildPositions();
    }
  }

  // Canvas size (center shorter layers vertically)
  const maxLayer = layersArr.length ? layersArr.length - 1 : 0;
  const maxRows = Math.max(1, ...layersArr.map((arr) => arr.length));
  const width = margin * 2 + (maxLayer + 1) * colGap + nodeW;
  const height = margin * 2 + maxRows * rowGap + nodeH;

  const coord = new Map<number, { x: number; y: number }>();
  layersArr.forEach((arr, layer) => {
    const offsetY = (maxRows - arr.length) * (rowGap / 2);
    arr.forEach((idx, k) => {
      const x = margin + layer * colGap;
      const y = margin + offsetY + k * rowGap;
      coord.set(idx, { x, y });
    });
  });

  return { width, height, nodeW, nodeH, coord };
}

const WorkflowSvg: React.FC<{
  parsed: KNL07ParseResult;
  onOpenNode: (idx: number) => void;
}> = ({ parsed, onOpenNode }) => {
  const tasks = React.useMemo(
    () => (parsed.nodes.filter((n) => (n as any) instanceof (TaskToken as any)) as NodeToken[]),
    [parsed],
  );
  // layout spacing controls
  const [colGap, setColGap] = React.useState<number>(320);
  const [rowGap, setRowGap] = React.useState<number>(130);
  const fallbackLayout = React.useMemo(() => createL2RLayout(tasks, colGap, rowGap), [tasks, colGap, rowGap]);

  // ELK async layout
  const [elkLayout, setElkLayout] = React.useState<L2RLayout | null>(null);
  const [elkEdgePaths, setElkEdgePaths] = React.useState<string[] | null>(null);
  const [elkEdgePathsFlipped, setElkEdgePathsFlipped] = React.useState<string[] | null>(null);
  const [flipY, setFlipY] = React.useState<boolean>(true); // ユーザーの好み: 上下逆
  const [edgeStyle, setEdgeStyle] = React.useState<"curved" | "orthogonal">("curved");
  const [bendRatio, setBendRatio] = React.useState<number>(0.5); // 0..1 の範囲で曲げ位置
  React.useEffect(() => {
    let aborted = false;
    const elk = new (ELK as any)();
  const nodeW = 220;
  const nodeH = 64;
  const margin = 40;
  const hSpacing = Math.max(20, colGap - nodeW); // 水平方向のノード間距離
  const vSpacing = Math.max(20, rowGap - nodeH); // 垂直方向のノード間距離

    const graph = {
      id: "root",
      layoutOptions: {
        "elk.algorithm": "layered",
        "elk.direction": "RIGHT",
        "elk.layered.spacing.nodeNodeBetweenLayers": String(hSpacing),
        "elk.spacing.nodeNode": String(vSpacing),
        "elk.layered.nodePlacement.strategy": "LINEAR_SEGMENT",
        "elk.edgeRouting": edgeStyle === "orthogonal" ? "ORTHOGONAL" : "SPLINES",
      },
      children: tasks
        .filter((n) => typeof n.nodeindex === "number")
        .map((n) => ({ id: String(n.nodeindex), width: nodeW, height: nodeH })),
      edges: tasks.flatMap((n) =>
        (n.to || [])
          .filter((t) => tasks.some((m) => m.nodeindex === t))
          .map((t) => ({ id: `${n.nodeindex}-${t}` , sources: [String(n.nodeindex)], targets: [String(t)] })),
      ),
    } as any;

    elk
      .layout(graph)
      .then((res: any) => {
        if (aborted || !res?.children) return;
        const coord = new Map<number, { x: number; y: number }>();
        const cx = new Map<string, { x: number; y: number }>();
        res.children.forEach((c: any) => {
          cx.set(c.id, { x: c.x, y: c.y });
          coord.set(Number(c.id), { x: margin + (c.x || 0), y: margin + (c.y || 0) });
        });
        // size
        const maxX = Math.max(...res.children.map((c: any) => (c.x || 0) + (c.width || nodeW)), 0);
        const maxY = Math.max(...res.children.map((c: any) => (c.y || 0) + (c.height || nodeH)), 0);
        const width = margin * 2 + maxX;
        const height = margin * 2 + maxY;
        setElkLayout({ width, height, nodeW, nodeH, coord });

        // Build polyline paths from ELK edge sections (add margin offset)
        const paths: string[] = [];
        const pathsFlip: string[] = [];
        (res.edges || []).forEach((e: any) => {
          (e.sections || []).forEach((s: any) => {
            const points: Array<{ x: number; y: number }> = [];
            if (s.startPoint) points.push(s.startPoint);
            if (Array.isArray(s.bendPoints)) points.push(...s.bendPoints);
            if (s.endPoint) points.push(s.endPoint);
            if (points.length < 2) return;
            const d = `M ${points.map((p: any) => `${margin + (p.x || 0)} ${margin + (p.y || 0)}`).join(" L ")}`;
            paths.push(d);
            // flipped Y
            const contentH = Math.max(0, height - margin * 2);
            const dFlip = `M ${points
              .map((p: any) => {
                const y = margin + (contentH - (p.y || 0));
                const x = margin + (p.x || 0);
                return `${x} ${y}`;
              })
              .join(" L ")}`;
            pathsFlip.push(dFlip);
          });
        });
        setElkEdgePaths(paths);
        setElkEdgePathsFlipped(pathsFlip);
      })
      .catch(() => {
        if (!aborted) {
          setElkLayout(null);
          setElkEdgePaths(null);
          setElkEdgePathsFlipped(null);
        }
      });
    return () => {
      aborted = true;
    };
  }, [tasks, colGap, rowGap, edgeStyle]);

  // 上下反転レイアウトへ変換
  const flipLayoutY = React.useCallback((base: L2RLayout): L2RLayout => {
    const margin = 40; // layout関数の既定マージンと一致
    const contentH = Math.max(0, base.height - margin * 2);
    const coord = new Map<number, { x: number; y: number }>();
    base.coord.forEach((p, k) => {
      const yRel = p.y - margin; // 0..contentH-nodeH のはず
      const y = margin + (contentH - yRel - base.nodeH);
      coord.set(k, { x: p.x, y });
    });
    return { ...base, coord };
  }, []);

  const baseLayout = elkLayout || fallbackLayout;
  const layout = React.useMemo(() => (flipY ? flipLayoutY(baseLayout) : baseLayout), [baseLayout, flipY, flipLayoutY]);

  // 入出力のスロット計算（重なり低減用）
  const slotInfo = React.useMemo(() => {
    const inMap = new Map<number, { order: number[]; pos: Map<number, number> }>();
    const outMap = new Map<number, { order: number[]; pos: Map<number, number> }>();
    const nodeY = (idx: number) => (layout.coord.get(idx)?.y ?? 0) + layout.nodeH / 2;

    // incoming slots (target側) — 送信元のYで安定ソート
    tasks.forEach((t) => {
      (t.to || []).forEach((v) => {
        if (!layout.coord.has(t.nodeindex!) || !layout.coord.has(v)) return;
        const list = inMap.get(v) || { order: [] as number[], pos: new Map<number, number>() };
        list.order.push(t.nodeindex!);
        inMap.set(v, list);
      });
    });
    inMap.forEach((entry, v) => {
      entry.order.sort((a, b) => nodeY(a) - nodeY(b));
      entry.order.forEach((src, i) => entry.pos.set(src, i));
    });

    // outgoing slots (source側) — 受信先のYで安定ソート
    tasks.forEach((s) => {
      const outs = (s.to || []).filter((v) => layout.coord.has(v));
      const order = [...outs].sort((a, b) => nodeY(a) - nodeY(b));
      const pos = new Map<number, number>();
      order.forEach((t, i) => pos.set(t, i));
      if (outs.length) outMap.set(s.nodeindex!, { order, pos });
    });

    return { inMap, outMap };
  }, [tasks, layout]);

  return (
    <div style={{ width: "100%", overflow: "auto", border: "1px solid #e5e7eb", borderRadius: 12 }}>
      <div style={{ padding: "8px 8px 0", display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
        <label style={{ fontSize: 12, color: "#475569", display: "inline-flex", gap: 8, alignItems: "center" }}>
          <input type="checkbox" checked={flipY} onChange={(e) => setFlipY(e.target.checked)} /> 上下反転
        </label>
        <div style={{ fontSize: 12, color: "#475569", display: "inline-flex", gap: 8, alignItems: "center" }}>
          エッジ:
          <label style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
            <input type="radio" name="edgeStyle" checked={edgeStyle === "curved"} onChange={() => setEdgeStyle("curved")} /> 曲線
          </label>
          <label style={{ display: "inline-flex", gap: 4, alignItems: "center" }}>
            <input type="radio" name="edgeStyle" checked={edgeStyle === "orthogonal"} onChange={() => setEdgeStyle("orthogonal")} /> 角線
          </label>
        </div>
        <div style={{ fontSize: 12, color: "#475569", display: "inline-flex", gap: 8, alignItems: "center" }}>
          列間隔:
          <input type="number" value={colGap} min={200} max={800} step={20} onChange={(e) => setColGap(Number(e.target.value) || 320)} style={{ width: 88 }} />
          行間隔:
          <input type="number" value={rowGap} min={100} max={600} step={10} onChange={(e) => setRowGap(Number(e.target.value) || 130)} style={{ width: 88 }} />
          曲げ係数:
          <input type="number" value={bendRatio} min={0.2} max={0.9} step={0.1} onChange={(e) => setBendRatio(Math.min(0.9, Math.max(0.2, Number(e.target.value))))} style={{ width: 72 }} />
        </div>
      </div>
      <svg width={layout.width} height={layout.height} style={{ display: "block" }}>
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
            <path d="M0,0 L9,3 L0,6 z" fill="#94a3b8" />
          </marker>
        </defs>
        {/* edges: prefer ELK paths (交差最小化)。flip時は反転済みパスを使用。*/}
        {elkLayout && (flipY ? elkEdgePathsFlipped : elkEdgePaths)?.map((d, i) => (
          <path key={`elk-e-${i}`} d={d} stroke="#94a3b8" fill="none" strokeWidth={2} markerEnd="url(#arrow)" />
        ))}

        {/* fallback edges when ELK not available */}
        {!elkLayout && tasks.flatMap((n) => {
          return (n.to || []).map((toIdx, i) => {
            if (!layout.coord.has(n.nodeindex!) || !layout.coord.has(toIdx)) return null;
            const a = layout.coord.get(n.nodeindex!)!;
            const b = layout.coord.get(toIdx)!;

            // slot positions
            const inEntry = slotInfo.inMap.get(toIdx);
            const outEntry = slotInfo.outMap.get(n.nodeindex!);
            const inCount = inEntry?.order.length ?? 1;
            const outCount = outEntry?.order.length ?? 1;
            const inIdx = inEntry?.pos.get(n.nodeindex!) ?? 0;
            const outIdx = outEntry?.pos.get(toIdx) ?? 0;

            const slotY = (base: number, count: number, idx: number) => {
              const frac = count > 1 ? (idx + 1) / (count + 1) : 0.5; // 0..1
              return base + layout.nodeH * frac;
            };
            const y1 = slotY(a.y, outCount, outIdx);
            const y2 = slotY(b.y, inCount, inIdx);
            const x1 = a.x + layout.nodeW;
            const x2 = b.x;
            const dx = Math.max(40, (x2 - x1) * bendRatio);

            let d = "";
            if (edgeStyle === "curved") {
              const c1x = x1 + dx, c1y = y1;
              const c2x = x2 - dx, c2y = y2;
              d = `M ${x1} ${y1} C ${c1x} ${c1y}, ${c2x} ${c2y}, ${x2} ${y2}`;
            } else {
              const mid1x = x1 + dx;
              const mid2x = x2 - dx;
              d = `M ${x1} ${y1} L ${mid1x} ${y1} L ${mid2x} ${y2} L ${x2} ${y2}`;
            }

            return (
              <path
                key={`e-${n.nodeindex}-${toIdx}-${i}`}
                d={d}
                stroke="#94a3b8"
                fill="none"
                strokeWidth={2}
                markerEnd="url(#arrow)"
              />
            );
          });
        })}

        {/* nodes */}
        {tasks.map((n) => {
          const p = layout.coord.get(n.nodeindex!);
          if (!p) return null;
          const attrs = n.attributes as any;
          const persona = attrs?.persona || "";
          const term = attrs?.outputOff ? 0.55 : 1;
          return (
            <g
              key={n.id}
              transform={`translate(${p.x},${p.y})`}
              style={{ cursor: "pointer", opacity: term }}
              onClick={() => n.nodeindex != null && onOpenNode(n.nodeindex)}
            >
              <rect width={layout.nodeW} height={layout.nodeH} rx={10} ry={10} fill="#ffffff" stroke="#CBD5E1" />
              <text x={10} y={18} fontSize={11} fill="#6b7280">{`NODE[${n.nodeindex}]`}</text>
              <text x={10} y={34} fontSize={13} fill="#111827">
                {n.value}
              </text>
              {persona ? (
                <text x={10} y={50} fontSize={11} fill="#64748b">
                  @{persona}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
};

/* =============== Node Detail / Dict Editor =============== */
// [Spec] U-3a TaskDictFields
// Purpose: タスク辞書エントリを編集するフォーム群。NodeDetailCard配下で使用。
// Inputs: フィールド文字列とsetter群、dictTaskKey。
// Outputs: onChangeイベントで親state更新。
// Notes: 単純なラベル付き2カラムグリッド。保存処理は親(NodeDetailCard)が担当。
type DictTask = NonNullable<KNL07Knowledge["tasks"]>[string];
const TaskDictFields: React.FC<{
  dictTaskKey?: string;
  name: string;
  setName: (v: string) => void;
  description: string;
  setDescription: (v: string) => void;
  focus: string;
  setFocus: (v: string) => void;
  scope: string;
  setScope: (v: string) => void;
  constraints: string;
  setConstraints: (v: string) => void;
  inputs: string;
  setInputs: (v: string) => void;
  outputs: string;
  setOutputs: (v: string) => void;
  aliases: string;
  setAliases: (v: string) => void;
  notes: string;
  setNotes: (v: string) => void;
}> = (p) => (
  <div style={{ display: "grid", gridTemplateColumns: "120px 1fr", gap: 8 }}>
    <label>Key</label>
    <input value={p.dictTaskKey || ""} readOnly />
    <label>name</label>
    <input value={p.name} onChange={(e) => p.setName(e.target.value)} />
    <label>description</label>
    <input value={p.description} onChange={(e) => p.setDescription(e.target.value)} />
    <label>focus</label>
    <input value={p.focus} onChange={(e) => p.setFocus(e.target.value)} />
    <label>scope</label>
    <input value={p.scope} onChange={(e) => p.setScope(e.target.value)} />
    <label>constraints</label>
    <input value={p.constraints} placeholder="カンマ区切り" onChange={(e) => p.setConstraints(e.target.value)} />
    <label>inputs</label>
    <input value={p.inputs} placeholder="カンマ区切り" onChange={(e) => p.setInputs(e.target.value)} />
    <label>outputs</label>
    <input value={p.outputs} placeholder="カンマ区切り" onChange={(e) => p.setOutputs(e.target.value)} />
    <label>aliases</label>
    <input value={p.aliases} placeholder="カンマ区切り" onChange={(e) => p.setAliases(e.target.value)} />
    <label>notes</label>
    <input value={p.notes} onChange={(e) => p.setNotes(e.target.value)} />
  </div>
);

// [Spec] U-3b NodeDetailCard
// Purpose: ノード個別の編集/詳細/プロンプト/辞書タブを統合するモーダル。
// Inputs: NodeToken、DSL全文、Tokenizer結果、編集コールバック、辞書情報、継承/位置情報。
// Outputs: DSLフラグメント更新、辞書保存、Promptプレビュー。
// Notes: タブ切替で edit/detail/prompt/dict を提供。DSL変更時はsnapshot付きでonEditDslを呼ぶ前提。
const NodeDetailCard: React.FC<{
  node: NodeToken;
  dsl: string;
  tokens: KNLToken[];
  onEditDsl: (newDsl: string) => void;
  onClose: () => void;
  getPrompt: () => string;
  personaSystem?: string;
  inheritance?: KNL07InheritanceIndex;
  nodePositions?: Record<number, { row: number; stepInRow: number }>;
  dictTaskKey?: string;
  dictTaskValue?: DictTask | undefined;
  onSaveDictTask?: (key: string, value: DictTask) => void;
}> = ({
  node,
  dsl,
  tokens,
  onEditDsl,
  onClose,
  getPrompt,
  personaSystem,
  inheritance,
  nodePositions,
  dictTaskKey,
  dictTaskValue,
  onSaveDictTask,
}) => {
  const prevToken = typeof node.tokenIndex === "number" && node.tokenIndex > 0 ? tokens[node.tokenIndex - 1] : null;
  const canEditOp = !!(prevToken instanceof OperatorToken && !(prevToken instanceof (DownArrowToken as any)));
  const curOp = canEditOp ? prevToken!.value.trim().charAt(0) : "";
  const oldStart = node.dslStart ?? node.position;
  const oldEnd = node.dslEnd ?? node.position + node.value.length;
  const oldDslPart = dsl.slice(oldStart, oldEnd);

  const attrs = oldDslPart
    ? {
        desc: (oldDslPart.match(/"([^"]*)"/) || [])[1] || "",
        // 複数@対応
        assign: Array.from(oldDslPart.matchAll(/@([\w\u3000-\u9FFF]+)/g))
          .map((m) => m[1])
          .join(", "),
        eq: (oldDslPart.match(/=\$([A-Za-z0-9_\[\]\d]+(?:\[\]|\[\d+\])?)/) || [])[1] || "",
        inputs: Array.from(oldDslPart.matchAll(/\(\$([\w\u3000-\u9FFF]+)\)/g))
          .map((m) => m[1])
          .join(", "),
        reqs: Array.from(oldDslPart.matchAll(/\(!([^)]+)\)/g))
          .map((m) => m[1])
          .join(", "),
        loops: (oldDslPart.match(/×([1-3])/) || [])[1] || "",
        op: curOp || "",
        kkey: (oldDslPart.match(/#([A-Za-z0-9_\-\.]+)/) || [])[1] || "",
      }
    : { desc: "", assign: "", eq: "", inputs: "", reqs: "", loops: "", op: "", kkey: "" };

  const [tab, setTab] = React.useState<"edit" | "detail" | "prompt" | "dict">("edit");
  const [val, setVal] = React.useState(node.value.replace(/^\^\??/, ""));
  const [undecided, setUndecided] = React.useState(node.type === "UNDECIDED_TASK");
  const [op, setOp] = React.useState(attrs.op);
  const [desc, setDesc] = React.useState(attrs.desc);
  const [assign, setAssign] = React.useState(attrs.assign);
  const [eq, setEq] = React.useState(attrs.eq);
  const [ins, setIns] = React.useState(attrs.inputs);
  const [reqs, setReqs] = React.useState(attrs.reqs);
  const [loops, setLoops] = React.useState(attrs.loops);
  const [kkey, setKkey] = React.useState(attrs.kkey);

  const [dtName, setDtName] = React.useState(dictTaskValue?.name || "");
  const [dtDesc, setDtDesc] = React.useState(dictTaskValue?.description || "");
  const [dtFocus, setDtFocus] = React.useState(dictTaskValue?.focus || "");
  const [dtScope, setDtScope] = React.useState(dictTaskValue?.scope || "");
  const [dtConstraints, setDtConstraints] = React.useState((dictTaskValue?.constraints || []).join(", "));
  const [dtInputs, setDtInputs] = React.useState((dictTaskValue?.inputs || []).join(", "));
  const [dtOutputs, setDtOutputs] = React.useState((dictTaskValue?.outputs || []).join(", "));
  const [dtAliases, setDtAliases] = React.useState((dictTaskValue?.aliases || []).join(", "));
  const [dtNotes, setDtNotes] = React.useState(dictTaskValue?.notes || "");

  React.useEffect(() => {
    setDtName(dictTaskValue?.name || "");
    setDtDesc(dictTaskValue?.description || "");
    setDtFocus(dictTaskValue?.focus || "");
    setDtScope(dictTaskValue?.scope || "");
    setDtConstraints((dictTaskValue?.constraints || []).join(", "));
    setDtInputs((dictTaskValue?.inputs || []).join(", "));
    setDtOutputs((dictTaskValue?.outputs || []).join(", "));
    setDtAliases((dictTaskValue?.aliases || []).join(", "));
    setDtNotes(dictTaskValue?.notes || "");
  }, [dictTaskKey, dictTaskValue]);

  const build = () => {
    let s = "";
    if (canEditOp && op) s += op;
    s += (undecided ? "^?" : "^") + val;
    if (desc) s += ` "${desc}"`;
    if (assign) assign.split(/[,\s]+/).map((v) => v.trim()).filter(Boolean).forEach((v) => (s += ` @${v}`));
    if (eq) s += ` =$${eq}`;
    if (ins) ins.split(",").map((v) => v.trim()).filter(Boolean).forEach((v) => (s += ` ($${v.replace(/^\$+/, "")})`));
    if (reqs) reqs.split(",").map((v) => v.trim()).filter(Boolean).forEach((v) => (s += ` (!${v.replace(/^!+/, "")})`));
    if (kkey) s += ` #${kkey.trim()}`;
    if (loops) s += ` ×${String(loops).replace(/[^1-3]/g, "")}`;
    return s.trim();
  };
  const preview = build();
  const saveDsl = () => {
    const next = dsl.slice(0, oldStart) + preview + dsl.slice(oldEnd);
    onEditDsl(next);
    onClose();
  };

  const pos = nodePositions?.[node.nodeindex ?? -1];
  const allNodes = (tokens.filter((t) => t instanceof NodeToken) as NodeToken[]) || [];
  const parents = (inheritance?.[node.nodeindex ?? -1] || []).map(
    (idx) => allNodes.find((n) => n.nodeindex === idx)?.value || `Node[${idx}]`,
  );

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div onClick={(e) => e.stopPropagation()} className="modal-panel">
        <div className="modal-header">
          <h4 className="m-0">Node Editor</h4>
          <button onClick={onClose} className="btn-icon">×</button>
        </div>

        <div style={{ display: "flex", gap: 8, marginBottom: 12, flexWrap: "wrap" }}>
          {(["edit", "detail", "prompt", "dict"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              style={{
                padding: "6px 12px",
                borderRadius: 6,
                border: tab === t ? "2px solid #3498db" : "1px solid #ccc",
                background: tab === t ? "#e8f4ff" : "#fff",
                fontWeight: "bold",
              }}
            >
              {t}
            </button>
          ))}
        </div>

        {tab === "edit" && (
          <>
            <div style={{ marginBottom: 12 }}>
              <label><b>Task</b></label>
              <div style={{ display: "flex", gap: 8, marginTop: 6 }}>
                <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <input type="checkbox" checked={undecided} onChange={(e) => setUndecided(e.target.checked)} />
                  <span>未定(^?)</span>
                </label>
                <input
                  value={val}
                  onChange={(e) => setVal(e.target.value)}
                  style={{ flex: 1, fontSize: 15, padding: 6, borderRadius: 6, border: "1px solid #ccc" }}
                />
              </div>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <div style={{ flex: 1, marginBottom: 12 }}>
                <label><b>desc</b></label>
                <input value={desc} onChange={(e) => setDesc(e.target.value)} className="input" />
              </div>
              <div style={{ flex: 1, marginBottom: 12 }}>
                <label><b>@Persona / @Tool（カンマ区切り）</b></label>
                <input value={assign} onChange={(e) => setAssign(e.target.value)} placeholder="editor, memo, library" className="input" />
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <div style={{ flex: 1, minWidth: 200, marginBottom: 12 }}>
                <label><b>=$out</b></label>
                <input value={eq} onChange={(e) => setEq(e.target.value)} placeholder="result / pages[]" className="input" />
              </div>
              <div style={{ flex: 2, minWidth: 240, marginBottom: 12 }}>
                <label><b>($in)</b></label>
                <input value={ins} onChange={(e) => setIns(e.target.value)} placeholder="a, b" className="input" />
              </div>
              <div style={{ flex: 2, minWidth: 240, marginBottom: 12 }}>
                <label><b>Requirements (!)</b></label>
                <input value={reqs} onChange={(e) => setReqs(e.target.value)} placeholder="基準1, 基準2" className="input" />
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <div style={{ width: 160, marginBottom: 12 }}>
                <label><b>#Key</b></label>
                <input value={kkey} onChange={(e) => setKkey(e.target.value)} placeholder="research" className="input" />
              </div>
              <div style={{ width: 160, marginBottom: 12 }}>
                <label><b>op</b></label>
                <input value={op} onChange={(e) => setOp(e.target.value)} placeholder="|, &, >, §" className="input" disabled={!canEditOp} />
              </div>
              <div style={{ width: 160, marginBottom: 12 }}>
                <label><b>Loop ×n</b></label>
                <input value={loops} onChange={(e) => setLoops(e.target.value.replace(/[^1-3]/g, ""))} placeholder="2" className="input" />
              </div>
            </div>
            <div style={{ marginBottom: 12 }}>
              <b>変更前:</b>
              <pre
                style={{
                  background: "#fff",
                  padding: 8,
                  borderRadius: 6,
                  fontSize: 13,
                  border: "1px solid #ddd",
                  marginTop: 4,
                  whiteSpace: "pre-wrap",
                }}
              >
                {oldDslPart}
              </pre>
              <b>変更後:</b>
              <pre
                style={{
                  background: "#e8f4ff",
                  padding: 8,
                  borderRadius: 6,
                  fontSize: 13,
                  border: "1px solid #3498db",
                  marginTop: 4,
                  whiteSpace: "pre-wrap",
                }}
              >
                {preview}
              </pre>
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={saveDsl} className="btn btn-primary">保存</button>
              <button onClick={onClose} className="btn">キャンセル</button>
            </div>
          </>
        )}

        {tab === "detail" && (
          <div style={{ marginTop: 8 }}>
            <table
              style={{
                width: "100%",
                fontSize: 14,
                background: "#fff",
                borderRadius: 6,
                border: "1px solid #ddd",
              }}
            >
              <tbody>
                <tr>
                  <td><b>index</b></td>
                  <td>{node.nodeindex}</td>
                </tr>
                <tr>
                  <td><b>type</b></td>
                  <td>{node.type}:{(node as any).typeIndex}</td>
                </tr>
                <tr>
                  <td><b>value</b></td>
                  <td>{node.value}</td>
                </tr>
                <tr>
                  <td><b>position</b></td>
                  <td>{pos ? `R${pos.row}, Step ${pos.stepInRow}` : "-"}</td>
                </tr>
                <tr>
                  <td><b>inherits</b></td>
                  <td>{parents.length ? parents.join(" -> ") : "-"}</td>
                </tr>
                <tr>
                  <td><b>from</b></td>
                  <td>{node.from.length ? node.from.join(", ") : "-"}</td>
                </tr>
                <tr>
                  <td><b>to</b></td>
                  <td>{node.to.length ? node.to.join(", ") : "-"}</td>
                </tr>
                <tr>
                  <td><b>explicit in</b></td>
                  <td>{node.explicitInputs.join(", ") || "-"}</td>
                </tr>
                <tr>
                  <td><b>explicit out</b></td>
                  <td>{node.explicitOutputs.join(", ") || "-"}</td>
                </tr>
                <tr>
                  <td><b>requirements</b></td>
                  <td>{node.explicitRequirements.join(", ") || "-"}</td>
                </tr>
                <tr>
                  <td><b>DSL fragment</b></td>
                  <td>
                    <pre
                      style={{
                        background: "#f8f8f8",
                        padding: 6,
                        borderRadius: 4,
                        whiteSpace: "pre-wrap",
                      }}
                    >
                      {node.dslRaw}
                    </pre>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {tab === "prompt" && (
          <div style={{ marginTop: 8 }}>
            {personaSystem ? (
              <div style={{ marginBottom: 8, fontSize: 12, color: "#6b7280" }}>
                <b>System (Persona):</b>
                <pre
                  style={{
                    background: "#fff",
                    padding: 8,
                    borderRadius: 6,
                    border: "1px solid #ddd",
                    whiteSpace: "pre-wrap",
                  }}
                >
                  {personaSystem}
                </pre>
              </div>
            ) : null}
            <pre
              style={{
                background: "#fff",
                padding: 8,
                borderRadius: 6,
                border: "1px solid #ddd",
                whiteSpace: "pre-wrap",
                maxHeight: 360,
                overflowY: "auto",
              }}
            >
              {getPrompt()}
            </pre>
          </div>
        )}

        {tab === "dict" && onSaveDictTask && (
          <div
            style={{
              background: "#fff",
              border: "1px solid #ddd",
              borderRadius: 8,
              padding: 12,
              marginTop: 8,
            }}
          >
            <TaskDictFields
              dictTaskKey={dictTaskKey}
              name={dtName}
              setName={setDtName}
              description={dtDesc}
              setDescription={setDtDesc}
              focus={dtFocus}
              setFocus={setDtFocus}
              scope={dtScope}
              setScope={setDtScope}
              constraints={dtConstraints}
              setConstraints={setDtConstraints}
              inputs={dtInputs}
              setInputs={setDtInputs}
              outputs={dtOutputs}
              setOutputs={setDtOutputs}
              aliases={dtAliases}
              setAliases={setDtAliases}
              notes={dtNotes}
              setNotes={setDtNotes}
            />
            <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
              <button
                onClick={() => {
                  const list = (s: string) => s.split(",").map((x) => x.trim()).filter(Boolean);
                  const val: DictTask = {
                    name: dtName || undefined,
                    description: dtDesc || undefined,
                    focus: dtFocus || undefined,
                    scope: dtScope || undefined,
                    constraints: list(dtConstraints),
                    inputs: list(dtInputs),
                    outputs: list(dtOutputs),
                    aliases: list(dtAliases),
                    notes: dtNotes || undefined,
                  };
                  if (dictTaskKey) onSaveDictTask(dictTaskKey, val);
                }}
                className="btn btn-primary"
              >
                辞書を保存
              </button>
              <button onClick={onClose} className="btn">閉じる</button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

/* =============== Visualization helpers (DAG/ledgers) =============== */
function getDagNodes(parsed: KNL07ParseResult | null) {
  if (!parsed) return [];
  const extractNodeSets = (c: any): any[] =>
    c.type === "NODE_SET" ? [c] : (c.children || []).flatMap((ch: any) => extractNodeSets(ch));
  const sets = parsed.executionTree.flatMap((it) => extractNodeSets(it as any));
  return parsed.nodes
    .filter((n) => n.type !== "PERSONA_NODE")
    .map((n) => {
      const ns = sets.find((s) => s.primaryNode?.id === n.id);
      const persona =
        (n.attributes as any)?.persona ||
        ns?.attachedNodes
          ?.filter((a: NodeToken) => a.type === "PERSONA_NODE")
          .map((a: NodeToken) => a.value)
          .join(", ") ||
        "";
      const binding = (n.attributes as any)?.binding?.key ? String((n.attributes as any).binding.key) : "";
      return {
        idx: n.nodeindex ?? -1,
        value: n.value,
        from: n.from,
        to: n.to,
        explicitInputs: n.explicitInputs,
        explicitOutputs: n.explicitOutputs,
        calculatedInputs: n.calculatedInputs,
        desc: (n.attributes as any)?.description ?? "",
        persona,
        binding,
      };
    });
}

function buildGlobalPortLedger(parsed: KNL07ParseResult | null) {
  if (!parsed) return [];
  const builderNodes = parsed.nodes.filter((node) => node.type !== "PERSONA_NODE");
  const builderNodeMap = new Map<number, NodeToken>();
  builderNodes.forEach((node) => {
    if (node.nodeindex != null) builderNodeMap.set(node.nodeindex, node);
  });

  const ledger = new Map<string, { producers: Set<string>; consumers: Set<string> }>();
  const ensure = (varKey: string) => {
    if (!ledger.has(varKey)) ledger.set(varKey, { producers: new Set(), consumers: new Set() });
    return ledger.get(varKey)!;
  };
  const encode = (idx: number | null | undefined) => (idx != null && idx >= 0 ? `node:${idx}` : "global");

  builderNodes.forEach((node) => {
    const idx = node.nodeindex ?? null;
    (node.explicitOutputs || []).forEach((out) => out && ensure(out).producers.add(encode(idx)));
    getNodeInputRecords(node).forEach((r) => {
      if (!r.input) return;
      const e = ensure(r.input);
      e.consumers.add(encode(idx));
      if (r.fromNode != null) e.producers.add(encode(r.fromNode));
    });
  });

  const decode = (token: string) =>
    token.startsWith("node:")
      ? (() => {
          const idx = Number(token.slice(5));
          const value = builderNodeMap.get(idx)?.value ?? "";
          return { token, idx, label: `Node[${idx}]${value ? ` ${value}` : ""}` };
        })()
      : { token, idx: null as number | null, label: "GLOBAL" };

  return Array.from(ledger.entries())
    .map(([variable, { producers, consumers }]) => ({
      variable,
      producers: Array.from(producers)
        .map(decode)
        .sort((a, b) => (a.idx ?? Number.MAX_SAFE_INTEGER) - (b.idx ?? Number.MAX_SAFE_INTEGER)),
      consumers: Array.from(consumers)
        .map(decode)
        .sort((a, b) => (a.idx ?? Number.MAX_SAFE_INTEGER) - (b.idx ?? Number.MAX_SAFE_INTEGER)),
    }))
    .sort((a, b) => a.variable.localeCompare(b.variable));
}

function buildFlowLedger(parsed: KNL07ParseResult | null) {
  if (!parsed) return [];
  const lines: string[] = [];
  parsed.nodes.forEach((node) => {
    const turn = node.nodeindex != null ? `turn${node.nodeindex}` : node.value;
    const label = node.value;
    getNodeInputRecords(node).forEach((r) => {
      if (!r.input) return;
      const fromNote = r.fromNode != null && r.fromNode >= 0 ? ` from Node[${r.fromNode}]` : "";
      const key = r.input.startsWith("$") ? r.input : `$${r.input}`;
      lines.push(`${turn}:select ${key} by ${label}${fromNote}`);
    });
    (node.explicitOutputs || []).forEach((out) => {
      const key = out.startsWith("$") ? out : `$${out}`;
      lines.push(`${turn}:make ${key} from ${label}`);
    });
  });
  return lines;
}

/* =============== Knowledge Tab（tool対応） =============== */
type DictKind = "task" | "var" | "persona" | "tool";
type AppearKind = "dict" | "dsl" | "llm";

const normalizeKey = (value: string) => (value || "").trim().toLowerCase();
function pickTaskKeyFromNodeValue(v: string) {
  return normalizeKey(v.replace(/^\^\??/, "").replace(/["@\(=].*$/, ""));
}
function collectDslRefs(parsed: KNL07ParseResult | null) {
  const result: {
    task: Record<string, number[]>;
    persona: Record<string, number[]>;
    var: Record<string, number[]>;
  } = { task: {}, persona: {}, var: {} };
  if (!parsed) return result;
  parsed.nodes.forEach((n) => {
    if ((n as any) instanceof (TaskToken as any)) {
      const key = pickTaskKeyFromNodeValue(n.value);
      if (key) (result.task[key] ||= []).push(n.nodeindex ?? -1);
      const p = (n.attributes as any)?.persona as string | undefined;
      if (p) {
        p.split(",")
          .map((s) => normalizeKey(s))
          .filter(Boolean)
          .forEach((one) => (result.persona[one] ||= []).push(n.nodeindex ?? -1));
      }
    }
    const set = new Set<string>();
    (n.explicitInputs || []).forEach((i) => set.add(i.replace(/^\$/, "")));
    (n.explicitOutputs || []).forEach((o) => set.add(KNL07Core.sanitizeVarKey(o)));
    Array.from(set).forEach((v) => (result.var[v] ||= []).push(n.nodeindex ?? -1));
  });
  return result;
}
function extractFirstJson<T = any>(text: string): T | null {
  const codeMatch = text.match(/```json([\s\S]*?)```/i);
  const source = codeMatch ? codeMatch[1] : text;
  const start = source.indexOf("{");
  const end = source.lastIndexOf("}");
  if (start === -1 || end === -1 || end <= start) {
    try {
      const aStart = source.indexOf("[");
      const aEnd = source.lastIndexOf("]");
      if (aStart >= 0 && aEnd > aStart) return JSON.parse(source.slice(aStart, aEnd + 1)) as T;
    } catch {}
    return null;
  }
  try {
    return JSON.parse(source.slice(start, end + 1)) as T;
  } catch {
    return null;
  }
}

// [Spec] U-4a KnowledgeTab
// Purpose: タスク/変数/ペルソナ/ツール辞書の参照・編集・LLM生成を統合。
// Inputs: state/dispatch、parsed結果、onOpenSettings。
// Outputs: 辞書state更新、LLM経由のエントリ生成、履歴localStorage管理。
// Notes: 左ペイン=フィルタ+リスト、右ペイン=フォーム+履歴。create/browseモード切替。
const KnowledgeTab: React.FC<{
  state: State;
  dispatch: React.Dispatch<Action>;
  parsed: KNL07ParseResult | null;
  onOpenSettings: () => void;
}> = ({ state, dispatch, parsed, onOpenSettings }) => {
  const [mode, setMode] = React.useState<"create" | "browse">("browse");
  const [typeFilter, setTypeFilter] = React.useState<Record<DictKind, boolean>>({
    task: true,
    var: true,
    persona: true,
    tool: true,
  });
  const [appearFilter, setAppearFilter] = React.useState<Record<AppearKind, boolean>>({
    dict: true,
    dsl: true,
    llm: false,
  });
  const [query, setQuery] = React.useState("");

  const refs = React.useMemo(() => collectDslRefs(parsed), [parsed]);

  type Item = { key: string; kind: DictKind; appear: AppearKind; count?: number };
  const leftItems: Item[] = React.useMemo(() => {
    const items: Item[] = [];
    const dict = (state.knowledge || {}) as AnyKnowledge;
    Object.keys(dict.tasks || {}).forEach((k) => items.push({ key: k, kind: "task", appear: "dict" }));
    Object.keys(dict.vars || {}).forEach((k) => items.push({ key: k, kind: "var", appear: "dict" }));
    Object.keys(dict.personas || {}).forEach((k) => items.push({ key: k, kind: "persona", appear: "dict" }));
    Object.keys(dict.tools || {}).forEach((k) => items.push({ key: k, kind: "tool", appear: "dict" }));
    Object.entries(refs.task).forEach(([k, ix]) => items.push({ key: k, kind: "task", appear: "dsl", count: ix.length }));
    Object.entries(refs.var).forEach(([k, ix]) => items.push({ key: k, kind: "var", appear: "dsl", count: ix.length }));
    Object.entries(refs.persona).forEach(([k, ix]) => {
      items.push({ key: k, kind: dict.tools?.[k] ? "tool" : "persona", appear: "dsl", count: ix.length });
    });
    return items
      .filter((it) => typeFilter[it.kind] && appearFilter[it.appear])
      .filter((it) => (query ? it.key.includes(query.toLowerCase()) : true))
      .sort((a, b) => a.key.localeCompare(b.key));
  }, [state.knowledge, refs, typeFilter, appearFilter, query]);

  const [selected, setSelected] = React.useState<Item | null>(null);

  // buffers
  const [taskBuf, setTaskBuf] = React.useState<DictTask>({
    name: "",
    description: "",
    focus: "",
    scope: "",
    constraints: [],
    inputs: [],
    outputs: [],
    aliases: [],
    notes: "",
  });
  const [personaBuf, setPersonaBuf] = React.useState<NonNullable<KNL07Knowledge["personas"]>[string]>({
    displayName: "",
    style: "",
    personality: "",
    policy: "",
    systemPrompt: "",
    aliases: [],
    notes: "",
  });
  const [varBuf, setVarBuf] = React.useState<NonNullable<KNL07Knowledge["vars"]>[string]>({
    name: "",
    type: "",
    description: "",
    examples: [],
    aliases: [],
    notes: "",
  });
  const [toolBuf, setToolBuf] = React.useState<ToolDef>({
    type: "",
    in: [],
    work: "",
    out: [],
    motif: "",
    aliases: [],
    notes: "",
  });

  // history for knowledge tab (local)
  type Hist = { ts: string; action: string; key: string; kind: DictKind; detail?: any };
  const [history, setHistory] = React.useState<Hist[]>(() => {
    try {
      return JSON.parse(localStorage.getItem("knl.knowledgeHist") || "[]");
    } catch {
      return [];
    }
  });
  const pushHist = (h: Hist) => {
    setHistory((prev) => {
      const next = [h, ...prev].slice(0, 200);
      localStorage.setItem("knl.knowledgeHist", JSON.stringify(next));
      return next;
    });
  };

  // load selected buffer
  React.useEffect(() => {
    if (!selected) return;
    const dict = (state.knowledge || {}) as AnyKnowledge;
    const k = normalizeKey(selected.key);
    if (selected.kind === "task") {
      const src = dict.tasks?.[k] || {};
      setTaskBuf({
        name: src.name || "",
        description: src.description || "",
        focus: src.focus || "",
        scope: src.scope || "",
        constraints: src.constraints || [],
        inputs: src.inputs || [],
        outputs: src.outputs || [],
        aliases: src.aliases || [],
        notes: src.notes || "",
      });
    } else if (selected.kind === "persona") {
      const src = dict.personas?.[k] || {};
      setPersonaBuf({
        displayName: src.displayName || "",
        style: src.style || "",
        personality: src.personality || "",
        policy: src.policy || "",
        systemPrompt: src.systemPrompt || "",
        aliases: src.aliases || [],
        notes: src.notes || "",
      });
    } else if (selected.kind === "var") {
      const src = dict.vars?.[k] || {};
      setVarBuf({
        name: src.name || "",
        type: src.type || "",
        description: src.description || "",
        examples: src.examples || [],
        aliases: src.aliases || [],
        notes: src.notes || "",
      });
    } else {
      const src = dict.tools?.[k] || {};
      setToolBuf({
        type: src.type || "",
        in: src.in || [],
        work: src.work || "",
        out: src.out || [],
        motif: src.motif || "",
        aliases: src.aliases || [],
        notes: src.notes || "",
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected?.key, selected?.kind]);

  const saveSelected = () => {
    if (!selected) return;
    const k = normalizeKey(selected.key);
    const dict: AnyKnowledge = JSON.parse(JSON.stringify(state.knowledge || {}));
    if (selected.kind === "task") {
      dict.tasks = dict.tasks || {};
      dict.tasks[k] = {
        name: taskBuf.name || undefined,
        description: taskBuf.description || undefined,
        focus: taskBuf.focus || undefined,
        scope: taskBuf.scope || undefined,
        constraints: (taskBuf.constraints || []).filter(Boolean),
        inputs: (taskBuf.inputs || []).filter(Boolean),
        outputs: (taskBuf.outputs || []).filter(Boolean),
        aliases: (taskBuf.aliases || []).filter(Boolean),
        notes: taskBuf.notes || undefined,
      };
    } else if (selected.kind === "persona") {
      dict.personas = dict.personas || {};
      dict.personas[k] = {
        displayName: personaBuf.displayName || undefined,
        style: personaBuf.style || undefined,
        personality: personaBuf.personality || undefined,
        policy: personaBuf.policy || undefined,
        systemPrompt: personaBuf.systemPrompt || undefined,
        aliases: (personaBuf.aliases || []).filter(Boolean),
        notes: personaBuf.notes || undefined,
      };
    } else if (selected.kind === "var") {
      dict.vars = dict.vars || {};
      dict.vars[k] = {
        name: varBuf.name || undefined,
        type: varBuf.type || undefined,
        description: varBuf.description || undefined,
        examples: (varBuf.examples || []).filter(Boolean),
        aliases: (varBuf.aliases || []).filter(Boolean),
        notes: varBuf.notes || undefined,
      };
    } else {
      dict.tools = dict.tools || {};
      dict.tools[k] = {
        type: toolBuf.type || undefined,
        in: (toolBuf.in || []).filter(Boolean),
        work: toolBuf.work || undefined,
        out: (toolBuf.out || []).filter(Boolean),
        motif: toolBuf.motif || undefined,
        aliases: (toolBuf.aliases || []).filter(Boolean),
        notes: toolBuf.notes || undefined,
      };
    }
    dispatch(applyKnowledgeAction(dict));
    pushHist({ ts: nowJP(), action: "save", key: k, kind: selected.kind, detail: dict });
  };

  // Create via LLM (uses global settings)
  type CreateBuf = { kind: DictKind; key: string; sample: string; schemaHint: string; comment: string };
  const [createBuf, setCreateBuf] = React.useState<CreateBuf>({
    kind: "task",
    key: "",
    sample: `{"name":"Research","description":"関連情報の収集と一次整理","focus":"関連情報の収集","scope":"対象領域の一次情報","constraints":["信頼できる出典"],"inputs":[],"outputs":["rawData"],"aliases":["調査","リサーチ"],"notes":""}`,
    schemaHint: `// kind: "task" | "persona" | "var" | "tool"
// tool: { "type"?: string, "in"?: string[], "work"?: string, "out"?: string[], "motif"?: string, "aliases"?: string[], "notes"?: string }
`,
    comment: "",
  });
  const [createLoading, setCreateLoading] = React.useState(false);
  const [createOutput, setCreateOutput] = React.useState<string>("");

  React.useEffect(() => {
    if (createBuf.kind === "tool") {
      setCreateBuf((p) => ({
        ...p,
        sample: `{"type":"memo","in":["notes[]"],"work":"要点をまとめる","out":["summary"],"motif":"メモ","aliases":["note","scratch"],"notes":""}`,
        schemaHint: `// tool: { "type"?: string, "in"?: string[], "work"?: string, "out"?: string[], "motif"?: string, "aliases"?: string[], "notes"?: string }`,
      }));
    } else if (createBuf.kind === "persona") {
      setCreateBuf((p) => ({
        ...p,
        sample: `{"displayName":"編集者","style":"明快・構造化","policy":"簡潔・一貫性","systemPrompt":"","aliases":["editor"],"notes":""}`,
        schemaHint: `// persona: { "displayName"?: string, "style"?: string, "personality"?: string, "policy"?: string, "systemPrompt"?: string, "aliases"?: string[], "notes"?: string }`,
      }));
    } else if (createBuf.kind === "var") {
      setCreateBuf((p) => ({
        ...p,
        sample: `{"name":"insights","type":"text","description":"分析から得た洞察","examples":[],"aliases":[],"notes":""}`,
        schemaHint: `// var: { "name"?: string, "type"?: string, "description"?: string, "examples"?: string[], "aliases"?: string[], "notes"?: string }`,
      }));
    } else {
      setCreateBuf((p) => ({
        ...p,
        sample: `{"name":"Research","description":"関連情報の収集と一次整理","focus":"関連情報の収集","scope":"対象領域の一次情報","constraints":["信頼できる出典"],"inputs":[],"outputs":["rawData"],"aliases":["調査","リサーチ"],"notes":""}`,
        schemaHint: `// task: { "name":string, "description":string, "focus"?:string, "scope"?:string, "constraints"?:string[], "inputs"?:string[], "outputs"?:string[], "aliases"?:string[], "notes"?:string }`,
      }));
    }
  }, [createBuf.kind]);

  const runCreate = async () => {
    if (!createBuf.key.trim()) {
      alert("キーを入力してください");
      return;
    }
    setCreateLoading(true);
    setCreateOutput("");
    try {
      const sys = [
        "You are a helpful dictionary assistant for a KNL workflow app.",
        "Return ONE JSON object that matches the requested schema and key.",
        "No extra prose. No code fences unless returning JSON code block.",
      ].join("\n");
      const prompt = [
        `# Target`,
        `- kind: ${createBuf.kind}`,
        `- key: ${normalizeKey(createBuf.key)}`,
        "",
        "# Global context",
        `- wish: ${state.globalWish || "(none)"}`,
        `- backbone: ${state.backbone || "(none)"}`,
        "",
        "# Sample entry (参考)",
        "```json",
        createBuf.sample.trim(),
        "```",
        "",
        "# Schema hint",
        "```",
        createBuf.schemaHint.trim(),
        "```",
        "",
        "# Notes",
        (createBuf.comment || "(none)").trim(),
        "",
        "Return only the JSON object for the entry (no explanations).",
      ].join("\n");

      const history = [{ role: "user" as const, content: prompt }];
      let text = "";
      const engine = (localStorage.getItem("knl.engine") as any) || "gemini";
      if (engine === "gemini") {
        const apiKey = localStorage.getItem("knl.apiKey") || "";
        text = await KNL07Core.callGeminiAPI({ history, apiKey, systemInstruction: sys });
      } else {
        const apiKey = localStorage.getItem("knl.claudeKey") || "";
        text = await KNL07Core.callClaudeAPI({ history, apiKey, systemPrompt: sys });
      }
      setCreateOutput(text);
      const obj = extractFirstJson<any>(text);
      if (!obj || typeof obj !== "object") {
        alert("JSONを抽出できませんでした。出力を確認してください。");
        return;
      }
      const k = normalizeKey(createBuf.key);
      const dict: AnyKnowledge = JSON.parse(JSON.stringify(state.knowledge || {}));
      if (createBuf.kind === "task") {
        dict.tasks = dict.tasks || {};
        dict.tasks[k] = obj;
      } else if (createBuf.kind === "persona") {
        dict.personas = dict.personas || {};
        dict.personas[k] = obj;
      } else if (createBuf.kind === "var") {
        dict.vars = dict.vars || {};
        dict.vars[k] = obj;
      } else {
        dict.tools = dict.tools || {};
        dict.tools[k] = obj;
      }
      dispatch(applyKnowledgeAction(dict));
      setSelected({ key: k, kind: createBuf.kind, appear: "dict" });
      pushHist({ ts: nowJP(), action: "create-llm", key: k, kind: createBuf.kind, detail: obj });
    } catch (e: any) {
      setCreateOutput(`Error: ${e?.message || String(e)}`);
    } finally {
      setCreateLoading(false);
    }
  };

  // UI
  return (
    <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 12 }}>
      <Card
        title={
          <div style={{ display: "flex", gap: 8 }}>
            <button
              className="chip-button"
              onClick={() => setMode("browse")}
              style={{
                background: mode === "browse" ? "#3498db" : "#fff",
                color: mode === "browse" ? "#fff" : "#111827",
                fontWeight: 700,
              }}
            >
              参照
            </button>
            <button
              className="chip-button"
              onClick={() => setMode("create")}
              style={{
                background: mode === "create" ? "#3498db" : "#fff",
                color: mode === "create" ? "#fff" : "#111827",
                fontWeight: 700,
              }}
            >
              新規登録
            </button>
          </div>
        }
        right={<button className="btn" onClick={onOpenSettings} title="LLM設定">{GEAR_ICON}</button>}
      >
        {mode === "browse" ? (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <label className="text-xs">検索</label>
              <input
                className="input"
                placeholder="keyを検索"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs">ノード種別</label>
              <div style={UIStyles.chipFilterRow}>
                {(["task", "var", "persona", "tool"] as DictKind[]).map((k) => (
                  <label key={k} className="chip-button" style={{ gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={typeFilter[k]}
                      onChange={(e) => setTypeFilter((p) => ({ ...p, [k]: e.target.checked }))}
                    />{" "}
                    {k}
                  </label>
                ))}
              </div>
            </div>
            <div>
              <label className="text-xs">登場種別</label>
              <div style={UIStyles.chipFilterRow}>
                {(["dict", "dsl", "llm"] as AppearKind[]).map((k) => (
                  <label key={k} className="chip-button" style={{ gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={appearFilter[k]}
                      onChange={(e) => setAppearFilter((p) => ({ ...p, [k]: e.target.checked }))}
                    />{" "}
                    {k}
                  </label>
                ))}
              </div>
            </div>
            <div className="bg-white border border-slate-200 rounded-lg p-2 max-h-[56vh] overflow-y-auto">
              {leftItems.map((it) => {
                const active = selected && it.key === selected.key && it.kind === selected.kind;
                return (
                  <div
                    key={`${it.appear}:${it.kind}:${it.key}`}
                    onClick={() => setSelected({ key: it.key, kind: it.kind, appear: it.appear })}
                    style={{
                      padding: "6px 8px",
                      borderRadius: 8,
                      cursor: "pointer",
                      marginBottom: 6,
                      background: active ? "#e0f2fe" : "#fff",
                      border: "1px solid #e5e7eb",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 8,
                    }}
                  >
                    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                      <span
                        style={{
                          fontSize: 11,
                          background: "#f3f4f6",
                          padding: "2px 6px",
                          borderRadius: 999,
                          color: "#374151",
                          minWidth: 54,
                          textAlign: "center",
                        }}
                      >
                        {it.kind}
                      </span>
                      <b>{it.key}</b>
                    </div>
                    <span style={{ fontSize: 11, color: "#6b7280" }}>
                      {it.appear}
                      {typeof it.count === "number" ? ` · ${it.count}` : ""}
                    </span>
                  </div>
                );
              })}
              {!leftItems.length && <div style={{ color: "#6b7280", fontSize: 12 }}>該当なし</div>}
            </div>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <div>
              <label className="text-xs">対象種別</label>
              <select
                className="select"
                value={createBuf.kind}
                onChange={(e) =>
                  setCreateBuf((p) => ({ ...p, kind: e.target.value as DictKind }))
                }
              >
                <option value="task">task</option>
                <option value="persona">persona</option>
                <option value="var">var</option>
                <option value="tool">tool</option>
              </select>
            </div>
            <div>
              <label className="text-xs">キー</label>
              <input
                className="input"
                placeholder="キー（正規化して保存されます）"
                value={createBuf.key}
                onChange={(e) => setCreateBuf((p) => ({ ...p, key: e.target.value }))}
              />
            </div>
            <div>
              <label className="text-xs">サンプル辞書</label>
              <textarea
                className="textarea font-mono"
                style={{ minHeight: 120 }}
                value={createBuf.sample}
                onChange={(e) => setCreateBuf((p) => ({ ...p, sample: e.target.value }))}
              />
            </div>
            <div>
              <label className="text-xs">型定義ヒント</label>
              <textarea
                className="textarea font-mono"
                style={{ minHeight: 100 }}
                value={createBuf.schemaHint}
                onChange={(e) => setCreateBuf((p) => ({ ...p, schemaHint: e.target.value }))}
              />
            </div>
            <div>
              <label className="text-xs">フリーコメント（補足）</label>
              <textarea
                className="textarea"
                style={{ minHeight: 80 }}
                value={createBuf.comment}
                onChange={(e) => setCreateBuf((p) => ({ ...p, comment: e.target.value }))}
              />
            </div>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="btn btn-primary" onClick={runCreate} disabled={createLoading}>
                {createLoading ? "生成中…" : "LLMで生成→登録"}
              </button>
              <button className="btn" onClick={() => setCreateOutput("")}>出力クリア</button>
            </div>
            {!!createOutput && (
              <div>
                <label className="text-xs">LLM出力</label>
                <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap">
                  {createOutput}
                </pre>
              </div>
            )}
          </div>
        )}
      </Card>

      {/* Right pane: 上=エディタ／下=履歴 */}
      <div style={{ display: "grid", gridTemplateRows: "1fr 220px", gap: 12 }}>
        <Card
          title={<h3 className="m-0">定義・編集</h3>}
          right={<button className="btn" onClick={onOpenSettings} title="LLM設定">{GEAR_ICON}</button>}
          style={{ minHeight: 300 }}
        >
          {!selected ? (
            <div style={{ color: "#6b7280" }}>左から項目を選ぶか、新規登録してください。</div>
          ) : selected.kind === "task" ? (
            <div style={UIStyles.twoColumnForm}>
              <label>name</label>
              <input className="input" value={taskBuf.name || ""} onChange={(e) => setTaskBuf((p) => ({ ...p, name: e.target.value }))} />
              <label>description</label>
              <input className="input" value={taskBuf.description || ""} onChange={(e) => setTaskBuf((p) => ({ ...p, description: e.target.value }))} />
              <label>focus</label>
              <input className="input" value={taskBuf.focus || ""} onChange={(e) => setTaskBuf((p) => ({ ...p, focus: e.target.value }))} />
              <label>scope</label>
              <input className="input" value={taskBuf.scope || ""} onChange={(e) => setTaskBuf((p) => ({ ...p, scope: e.target.value }))} />
              <label>constraints</label>
              <input className="input" placeholder="a, b" value={(taskBuf.constraints || []).join(", ")} onChange={(e) => setTaskBuf((p) => ({ ...p, constraints: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>inputs</label>
              <input className="input" placeholder="a, b" value={(taskBuf.inputs || []).join(", ")} onChange={(e) => setTaskBuf((p) => ({ ...p, inputs: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>outputs</label>
              <input className="input" placeholder="a, b" value={(taskBuf.outputs || []).join(", ")} onChange={(e) => setTaskBuf((p) => ({ ...p, outputs: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>aliases</label>
              <input className="input" placeholder="a, b" value={(taskBuf.aliases || []).join(", ")} onChange={(e) => setTaskBuf((p) => ({ ...p, aliases: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>notes</label>
              <input className="input" value={taskBuf.notes || ""} onChange={(e) => setTaskBuf((p) => ({ ...p, notes: e.target.value }))} />
              <div />
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-primary" onClick={saveSelected}>保存</button>
                <button className="btn" onClick={() => setSelected(null)}>クリア</button>
              </div>
            </div>
          ) : selected.kind === "persona" ? (
            <div style={UIStyles.twoColumnForm}>
              <label>displayName</label>
              <input className="input" value={personaBuf.displayName || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, displayName: e.target.value }))} />
              <label>style</label>
              <input className="input" value={personaBuf.style || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, style: e.target.value }))} />
              <label>personality</label>
              <input className="input" value={personaBuf.personality || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, personality: e.target.value }))} />
              <label>policy</label>
              <input className="input" value={personaBuf.policy || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, policy: e.target.value }))} />
              <label>systemPrompt</label>
              <input className="input" value={personaBuf.systemPrompt || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, systemPrompt: e.target.value }))} />
              <label>aliases</label>
              <input className="input" placeholder="a, b" value={(personaBuf.aliases || []).join(", ")} onChange={(e) => setPersonaBuf((p) => ({ ...p, aliases: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>notes</label>
              <input className="input" value={personaBuf.notes || ""} onChange={(e) => setPersonaBuf((p) => ({ ...p, notes: e.target.value }))} />
              <div />
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-primary" onClick={saveSelected}>保存</button>
                <button className="btn" onClick={() => setSelected(null)}>クリア</button>
              </div>
            </div>
          ) : selected.kind === "var" ? (
            <div style={UIStyles.twoColumnForm}>
              <label>name</label>
              <input className="input" value={varBuf.name || ""} onChange={(e) => setVarBuf((p) => ({ ...p, name: e.target.value }))} />
              <label>type</label>
              <input className="input" value={varBuf.type || ""} onChange={(e) => setVarBuf((p) => ({ ...p, type: e.target.value }))} />
              <label>description</label>
              <input className="input" value={varBuf.description || ""} onChange={(e) => setVarBuf((p) => ({ ...p, description: e.target.value }))} />
              <label>examples</label>
              <input className="input" placeholder='"a","b"' value={(varBuf.examples || []).join(", ")} onChange={(e) => setVarBuf((p) => ({ ...p, examples: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>aliases</label>
              <input className="input" placeholder="a, b" value={(varBuf.aliases || []).join(", ")} onChange={(e) => setVarBuf((p) => ({ ...p, aliases: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>notes</label>
              <input className="input" value={varBuf.notes || ""} onChange={(e) => setVarBuf((p) => ({ ...p, notes: e.target.value }))} />
              <div />
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-primary" onClick={saveSelected}>保存</button>
                <button className="btn" onClick={() => setSelected(null)}>クリア</button>
              </div>
            </div>
          ) : (
            <div style={UIStyles.twoColumnForm}>
              <label>type</label>
              <input className="input" placeholder="memo/library/furnace/scale/sign/knife ..." value={toolBuf.type || ""} onChange={(e) => setToolBuf((p) => ({ ...p, type: e.target.value }))} />
              <label>in</label>
              <input className="input" placeholder="a, b" value={(toolBuf.in || []).join(", ")} onChange={(e) => setToolBuf((p) => ({ ...p, in: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>work</label>
              <input className="input" placeholder="要点をまとめる / 混ぜる / 並べる / はかる / 切る ..." value={toolBuf.work || ""} onChange={(e) => setToolBuf((p) => ({ ...p, work: e.target.value }))} />
              <label>out</label>
              <input className="input" placeholder="x, y" value={(toolBuf.out || []).join(", ")} onChange={(e) => setToolBuf((p) => ({ ...p, out: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>motif</label>
              <input className="input" placeholder="メモ/本/炉/天秤/サイン/ナイフ ..." value={toolBuf.motif || ""} onChange={(e) => setToolBuf((p) => ({ ...p, motif: e.target.value }))} />
              <label>aliases</label>
              <input className="input" placeholder="a, b" value={(toolBuf.aliases || []).join(", ")} onChange={(e) => setToolBuf((p) => ({ ...p, aliases: e.target.value.split(",").map((x) => x.trim()).filter(Boolean) }))} />
              <label>notes</label>
              <input className="input" value={toolBuf.notes || ""} onChange={(e) => setToolBuf((p) => ({ ...p, notes: e.target.value }))} />
              <div />
              <div style={{ display: "flex", gap: 8 }}>
                <button className="btn btn-primary" onClick={saveSelected}>保存</button>
                <button className="btn" onClick={() => setSelected(null)}>クリア</button>
              </div>
            </div>
          )}
        </Card>

        <Card title={<h3 className="m-0">履歴</h3>}>
          <div className="bg-white border border-slate-200 rounded-lg p-2 text-xs max-h-[180px] overflow-y-auto">
            {history.length ? (
              history.map((h, i) => (
                <div key={i} className="border-b border-slate-100 py-1">
                  <span className="text-slate-500">[{h.ts}]</span>{" "}
                  <b>{h.action}</b> <span>{h.kind}</span> <code>{h.key}</code>
                </div>
              ))
            ) : (
              <div style={{ color: "#6b7280" }}>履歴はありません</div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};

/* =============== Execute/History（実行器と履歴ストア） =============== */
type RunMeta = { runId: string; name: string; startedAt: string; finishedAt?: string; dsl: string; wish?: string; backbone?: string; };
type HistoryEntry = {
  id: string; // `${runId}:${taskId}`
  runId: string;
  taskId: number;
  runName?: string;
  createdAt: string;
  value: {
    prompt: string;
    response: string;
    inputs: Record<string, any>;
    outputs: Record<string, any>;
    meta: { persona?: string; tools?: string[]; nodeLabel?: string; nodeIdx?: number; requiresApproval?: boolean; approved?: boolean; approvedAt?: string; approvalNote?: string };
  };
};

const LS_RUNS = "knl.exec.runs";
const LS_ENTRIES = "knl.exec.entries";
function loadRuns(): RunMeta[] { try { return JSON.parse(localStorage.getItem(LS_RUNS) || "[]"); } catch { return []; } }
function saveRuns(list: RunMeta[]) { localStorage.setItem(LS_RUNS, JSON.stringify(list)); }
function loadEntries(): HistoryEntry[] { try { return JSON.parse(localStorage.getItem(LS_ENTRIES) || "[]"); } catch { return []; } }
function saveEntries(list: HistoryEntry[]) { localStorage.setItem(LS_ENTRIES, JSON.stringify(list)); }
function upsertEntry(e: HistoryEntry) {
  const list = loadEntries();
  const i = list.findIndex((x) => x.id === e.id);
  if (i >= 0) list[i] = e; else list.unshift(e);
  saveEntries(list.slice(0, 5000));
}
function patchEntry(id: string, patch: Partial<HistoryEntry["value"]["meta"]>) {
  const list = loadEntries();
  const i = list.findIndex((x) => x.id === id);
  if (i >= 0) {
    list[i] = { ...list[i], value: { ...list[i].value, meta: { ...list[i].value.meta, ...patch } } };
    saveEntries(list);
  }
}

const isTask = (n: KNLToken): n is NodeToken => (n as any) instanceof (TaskToken as any);
const hasCheckGate = (n: NodeToken) => !!((n.attributes as any)?.check);
function topoOrder(nodes: NodeToken[]) {
  const map = new Map<number, NodeToken>();
  nodes.forEach((n) => { if (n.nodeindex != null) map.set(n.nodeindex, n); });
  const indeg = new Map<number, number>();
  const out = new Map<number, number[]>();
  map.forEach((_, idx) => { indeg.set(idx, 0); out.set(idx, []); });
  map.forEach((n, idx) => n.to.forEach((t) => {
    if (map.has(t)) {
      indeg.set(t, (indeg.get(t) || 0) + 1);
      out.get(idx)!.push(t);
    }
  }));
  const q: number[] = [];
  indeg.forEach((d, i) => d === 0 && q.push(i));
  const res: number[] = [];
  while (q.length) {
    const u = q.shift()!;
    res.push(u);
    (out.get(u) || []).forEach((v) => {
      const d = (indeg.get(v) || 0) - 1;
      indeg.set(v, d);
      if (d === 0) q.push(v);
    });
  }
  map.forEach((_, i) => { if (!res.includes(i)) res.push(i); });
  return res;
}
function collectAncestors(tasks: NodeToken[], nodeIdx: number): Set<number> {
  const set = new Set<number>();
  const map = new Map<number, NodeToken>();
  tasks.forEach((n) => { if (n.nodeindex != null) map.set(n.nodeindex, n); });
  const stack = [nodeIdx];
  while (stack.length) {
    const cur = stack.pop()!;
    const n = map.get(cur);
    if (!n) continue;
    for (const p of (n.from || [])) {
      if (!map.has(p)) continue;
      if (!set.has(p)) { set.add(p); stack.push(p); }
    }
  }
  return set;
}

function buildLineageHistoryMessages(
  runId: string,
  tasks: NodeToken[],
  nodeIdx: number,
  limitPairs = KNL07Flags.lineageMaxPairs,
): { messages: KNL07ConversationMessage[]; usedTaskIds: number[] } {
  const ancestors = collectAncestors(tasks, nodeIdx);
  if (!ancestors.size) return { messages: [], usedTaskIds: [] };

  // 既存の topoOrder で昇順に並び替え、先祖だけ抽出
  const order = topoOrder(tasks).filter((i) => ancestors.has(i));
  const entries = loadEntries(); // 新しい順で格納されている
  const pickLatest = (taskId: number) => entries.find((e) => e.runId === runId && e.taskId === taskId);

  const msgs: KNL07ConversationMessage[] = [];
  const used: number[] = [];
  for (const tid of order) {
    const e = pickLatest(tid);
    if (!e) continue;
    used.push(tid);
    // user: 送ったプロンプト全文
    msgs.push({ role: "user", content: e.value.prompt });
    // assistant: 生のレスポンス全文（未パース）
    msgs.push({ role: "assistant", content: e.value.response });
    if (Math.floor(msgs.length / 2) >= limitPairs) break;
  }
  return { messages: msgs, usedTaskIds: used };
}
/* 詳細モーダル（承認/リトライ/差戻し） */
type TaskRunStatus = "pending" | "running" | "await_approval" | "approved" | "done" | "error" | "skipped";
type ExecTaskRow = {
  idx: number; label: string; persona?: string; tools: string[];
  inputs: string[]; outputs: string[]; requiresApproval: boolean;
  status: TaskRunStatus; startedAt?: string; finishedAt?: string; error?: string;
};

// [Spec] U-2c ExecDetailModal
// Purpose: 実行タスクのI/O・承認操作・差戻しを行う詳細モーダル。
// Inputs: open/onClose、選択行(row)、最新IO、アクションcallbacks、predecessors。
// Outputs: approve/retry/sendBackなど親コールバックで実行。
// Notes: ExecuteTabから開かれ、承認メモや差戻し先選択を扱う。
const ExecDetailModal: React.FC<{
  open: boolean;
  onClose: () => void;
  row: ExecTaskRow | null;
  lastIO?: { prompt: string; response: string; inputs: Record<string, any>; outputs: Record<string, any>; system?: string };
  actions: {
    approve: (note?: string) => void;
    retry: () => void;
    sendBack: (toTaskId: number) => void;
    copy: () => void;
  };
  predecessors: number[];
}> = ({ open, onClose, row, lastIO, actions, predecessors }) => {
  const [note, setNote] = React.useState("");
  const [toId, setToId] = React.useState<number | "">(predecessors[0] ?? "");
  React.useEffect(() => { if (open) { setNote(""); setToId(predecessors[0] ?? ""); } }, [open, predecessors]);
  if (!open || !row) return null;
  return (
    <div className="modal-backdrop" onClick={onClose}><div className="modal-panel" onClick={(e) => e.stopPropagation()}>
      <div className="modal-header">
        <h4 className="m-0">実行詳細: Node[{row.idx}] {row.label}</h4>
        <button className="btn-icon" onClick={onClose}>×</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <b>Inputs</b>
          <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[200px] overflow-y-auto">
            {lastIO ? JSON.stringify(lastIO.inputs, null, 2) : "(なし)"}
          </pre>
        </div>
        <div>
          <b>Outputs</b>
          <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[200px] overflow-y-auto">
            {lastIO ? JSON.stringify(lastIO.outputs, null, 2) : "(なし)"}
          </pre>
        </div>
      </div>

      <div style={{ marginTop: 8 }}>
        <b>Prompt</b>
        <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[200px] overflow-y-auto">
          {lastIO?.prompt || "(なし)"}
        </pre>
      </div>
      <div style={{ marginTop: 8 }}>
        <b>Response</b>
        <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[200px] overflow-y-auto">
          {lastIO?.response || "(なし)"}
        </pre>
      </div>

      <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
        <div>
          <label className="text-xs">承認メモ（任意）</label>
          <input className="input" value={note} onChange={(e) => setNote(e.target.value)} placeholder="追記メモ / 補足" />
          <div style={{ display: "flex", gap: 8, marginTop: 8, flexWrap: "wrap" }}>
            <button className="btn btn-primary" onClick={() => actions.approve(undefined)}>承認</button>
            <button className="btn" onClick={() => actions.approve(note)} disabled={!note}>承認（メモ追記）</button>
            <button className="btn" onClick={actions.retry}>リトライ</button>
            <button className="btn" onClick={actions.copy}>I/Oをコピー</button>
          </div>
        </div>
        <div>
          <label className="text-xs">差戻し先（前工程へ戻す）</label>
          <div style={{ display: "flex", gap: 6 }}>
            <select className="select" value={toId} onChange={(e) => setToId(Number(e.target.value))}>
              {predecessors.map((pid) => (<option key={pid} value={pid}>Node[{pid}]</option>))}
            </select>
            <button className="btn" onClick={() => typeof toId === "number" && actions.sendBack(toId)} disabled={typeof toId !== "number"}>差戻し実行</button>
          </div>
          <div style={{ fontSize: 12, color: "#6b7280", marginTop: 6 }}>差戻し後、選択ノードから再度順次実行します。</div>
        </div>
      </div>
    </div></div>
  );
};

type RunStatus = "idle" | "running" | "stopped" | "completed" | "failed";

// [Spec] U-2d ExecuteTab
// Purpose: DSLトポ順実行とLLMリクエスト管理、履歴連携、承認フローを提供。
// Inputs: state (DSL/knowledgeなど)、parsed。
// Outputs: 実行行(rows)ステート、LLM呼び出し、履歴保存(localStorage)、承認ダイアログ制御。
// Notes: topoOrderから初期計画生成。Pause on check、差戻しなどランタイム制御を内包。
const ExecuteTab: React.FC<{ state: State; parsed: KNL07ParseResult | null; }> = ({ state, parsed }) => {
  const tasks = React.useMemo(() => (parsed?.nodes.filter(isTask) as NodeToken[]) || [], [parsed]);
  const makePlan = React.useCallback(() => {
    const order = topoOrder(tasks);
    return order.map((idx) => {
      const n = tasks.find((x) => x.nodeindex === idx)!;
      const atTokens = (n.dslRaw.match(/@([\w\u3000-\u9FFF]+)/g) || []).map((x) => x.replace(/^@/, ""));
      const tools = atTokens.filter((k) => !!state.knowledge?.tools?.[normalizeKey(k)]);
      const personas = atTokens.filter((k) => !state.knowledge?.tools?.[normalizeKey(k)]);
      const personaMerged = [((n.attributes as any)?.persona || ""), ...personas].filter(Boolean).join(",");
      return {
        idx, label: n.value, persona: personaMerged || undefined, tools,
        inputs: n.explicitInputs || [], outputs: n.explicitOutputs || [],
        requiresApproval: hasCheckGate(n), status: "pending" as TaskRunStatus,
      };
    });
  }, [tasks, state.knowledge]);

  const [rows, setRows] = React.useState<ExecTaskRow[]>(makePlan);
  const [cursor, setCursor] = React.useState(0);
  const [runStatus, setRunStatus] = React.useState<RunStatus>("idle");
  const [runName, setRunName] = React.useState(`run-${new Date().toISOString().replace(/[:.]/g, "-")}`);
  const [runId, setRunId] = React.useState<string>("");
  const [vars, setVars] = React.useState<Record<string, any>>({});
  const [tags] = React.useState<Record<string, any>>({});
  const [pauseOnCheck, setPauseOnCheck] = React.useState(true);
  const stopRef = React.useRef(false);

  // 詳細モーダル
  const [detailOpen, setDetailOpen] = React.useState(false);
  const [detailRow, setDetailRow] = React.useState<ExecTaskRow | null>(null);
  const [detailIO, setDetailIO] = React.useState<{ prompt: string; response: string; inputs: Record<string, any>; outputs: Record<string, any>; system?: string }>();

  // DSL変更（^?追加など）で計画を再構築/マージ
  React.useEffect(() => {
    const nextPlan = makePlan();
    setRows((prev) => {
      const map = new Map(prev.map((r) => [r.idx, r]));
      const merged: ExecTaskRow[] = nextPlan.map((r) => {
        const prevRow = map.get(r.idx);
        if (!prevRow) return r;
        // 新しい計画のセマンティック値を優先し、ランタイムのみ旧値を保持
        return {
          ...r,
          status: prevRow.status,
          startedAt: prevRow.startedAt,
          finishedAt: prevRow.finishedAt,
          error: prevRow.error,
        };
      });
      return merged;
    });
  }, [makePlan]);

  // ラン作成
  const ensureRun = React.useCallback(() => {
    if (runId) return runId;
    const id = `r_${Date.now()}`;
    setRunId(id);
    const runs = loadRuns();
    runs.unshift({ runId: id, name: runName || id, startedAt: new Date().toISOString(), dsl: state.dsl, wish: state.globalWish, backbone: state.backbone });
    saveRuns(runs.slice(0, 300));
    return id;
  }, [runId, runName, state.dsl, state.globalWish, state.backbone]);

  async function callLLM(prompt: string, system?: string) {
    const engine = (localStorage.getItem("knl.engine") as any) || "gemini";
    const history = [{ role: "user" as const, content: prompt }];
    if (engine === "gemini") {
      const apiKey = localStorage.getItem("knl.apiKey") || "";
      return await KNL07Core.callGeminiAPI({ history, apiKey, systemInstruction: system });
    } else {
      const apiKey = localStorage.getItem("knl.claudeKey") || "";
      return await KNL07Core.callClaudeAPI({ history, apiKey, systemPrompt: system });
    }
  }
  function buildToolHints(toolKeys: string[]) {
    const k = state.knowledge;
    const lines: string[] = [];
    toolKeys.forEach((tk) => {
      const t = k?.tools?.[normalizeKey(tk)];
      if (!t) return;
      lines.push(`- tool:${tk}${t.motif ? ` motif=${t.motif}` : ""} type=${t.type || ""}`);
      if (t.in?.length) lines.push(`  in: ${t.in.join(", ")}`);
      if (t.work) lines.push(`  work: ${t.work}`);
      if (t.out?.length) lines.push(`  out: ${t.out.join(", ")}`);
    });
    return lines.join("\n");
  }
  function predecessorsOf(taskId: number) {
    const n = tasks.find((x) => x.nodeindex === taskId);
    return n?.from || [];
  }

  // 1タスク実行: 結果を返し、✓ゲート時に呼び出し元が即停止できるようにする
  async function runOne(rowIndex: number): Promise<"ok" | "await_approval" | "error"> {
  const row = rows[rowIndex];
  const node = tasks.find((n) => n.nodeindex === row.idx);
  if (!node) return "error";
  // ラベルは常に最新のノード値から導出（UIと同一ロジック）
  const liveLabel = node ? (extractTaskBaseName(node.value) || node.value) : row.label;

  // 入力収集（既存）
  const inputObj: Record<string, any> = {};
  (row.inputs || []).forEach((k) => { const key = k.replace(/^\$/, ""); inputObj[key] = vars[key]; });

  // System（既存）
  const system = [
    KNL07Core.buildPersonaInstruction((node.attributes as any) || {}),
    buildToolHints(row.tools),
  ].filter(Boolean).join("\n\n");

  // Prompt（既存）
  const prompt = KNL07Core.buildPrompt({
    node, allNodes: parsed?.nodes || [], ctx: { vars, tags },
    inheritance: KNL07Core.buildInheritanceIndex(parsed?.executionTree || []),
    global: state.globalWish, backbone: state.backbone, position: (parsed?.nodePositions || {})[row.idx],
  });

  setRows((R) => R.map((x, i) => (i === rowIndex ? { ...x, status: "running", startedAt: new Date().toISOString(), error: undefined } : x)));

  try {
    // 追加: 今の runId を必ず確定（先祖履歴参照に必要）
    const rid = ensureRun();

    // 追加: 依存系譜の履歴を構築（root→対象タスクの直前まで）
    let lineage: KNL07ConversationMessage[] = [];
    if (KNL07Flags.useLineageHistory) {
      lineage = buildLineageHistoryMessages(rid, tasks, row.idx).messages;
    }

    // 置換: callLLM ラッパーを使わず直接 API を呼ぶ（history に lineage + 現在の user を渡す）
    const engine = (localStorage.getItem("knl.engine") as any) || "gemini";
    const history: KNL07ConversationMessage[] = [...lineage, { role: "user", content: prompt }];

    let response = "";
    if (engine === "gemini") {
      const apiKey = localStorage.getItem("knl.apiKey") || "";
      response = await KNL07Core.callGeminiAPI({ history, apiKey, systemInstruction: system });
    } else {
      const apiKey = localStorage.getItem("knl.claudeKey") || "";
      response = await KNL07Core.callClaudeAPI({ history, apiKey, systemPrompt: system });
    }

    // 以下は既存: 応答解析→出力保存→HistoryEntry への保存→✓処理
    const parsedLLM = KNL07Core.parseLLM(response);
    const outVars: Record<string, any> = {};
    (row.outputs || []).forEach((ok) => {
      const key = ok.replace(/^\$/, "");
      const v = parsedLLM.variables?.[key];
      outVars[key] = (v?.value ?? v) ?? parsedLLM.details ?? response;
    });
    const nodeResult = parsedLLM.summary || parsedLLM.details || response;
    setVars((V) => ({ ...V, ...outVars, [`node_${row.idx}_result`]: nodeResult }));

    const entry: HistoryEntry = {
      id: `${rid}:${row.idx}`, runId: rid, taskId: row.idx, runName,
      createdAt: new Date().toISOString(),
      value: {
        prompt, response, inputs: inputObj, outputs: outVars,
        meta: { persona: row.persona, tools: row.tools, nodeLabel: liveLabel, nodeIdx: row.idx, requiresApproval: row.requiresApproval, approved: !row.requiresApproval },
      },
    };
    upsertEntry(entry);

    if (row.requiresApproval) {
      setRows((R) => R.map((x, i) => (i === rowIndex ? { ...x, status: "await_approval", finishedAt: new Date().toISOString() } : x)));
      setDetailRow({ ...row, status: "await_approval" });
      setDetailIO({ prompt, response, inputs: inputObj, outputs: outVars, system });
      setDetailOpen(true);
      if (pauseOnCheck) { setRunStatus("stopped"); return "await_approval"; }
    } else {
      setRows((R) => R.map((x, i) => (i === rowIndex ? { ...x, status: "done", finishedAt: new Date().toISOString() } : x)));
    }
    return "ok";
  } catch (e: any) {
    setRows((R) => R.map((x, i) => (i === rowIndex ? { ...x, status: "error", error: e?.message || String(e), finishedAt: new Date().toISOString() } : x)));
    return "error";
  }
}

  async function runAll(fromIndex = 0) {
    if (!rows.length) return;
    setRunStatus("running"); stopRef.current = false;
    for (let i = fromIndex; i < rows.length; i++) {
      setCursor(i);
      if (stopRef.current) { setRunStatus("stopped"); break; }
      const r = rows[i];
      if (r.status === "done" || r.status === "approved") continue;
      const res = await runOne(i);
      if (res === "error") { setRunStatus("failed"); break; }
      if (res === "await_approval") break; // ✓で一時中断（即時停止）
    }
    if (!stopRef.current && runStatus !== "failed" && rows.every((x) => ["done","approved"].includes(x.status))) {
      setRunStatus("completed");
      const runs = loadRuns();
      const idx = runs.findIndex((r) => r.runId === runId);
      if (idx >= 0) { runs[idx].finishedAt = new Date().toISOString(); saveRuns(runs); }
    }
  }

  function resetRun() {
    setRunId("");
    setRows(makePlan());
    setVars({});
    setCursor(0);
    setRunStatus("idle");
  }

  // 承認/リトライ/差戻し
  const approve = (note?: string) => {
    if (!detailRow) return;
    const id = `${runId}:${detailRow.idx}`;
    patchEntry(id, { approved: true, approvedAt: new Date().toISOString(), approvalNote: note });
    if (note && (detailRow.outputs || []).length) {
      const key = detailRow.outputs[0].replace(/^\$/, "");
      setVars((V) => ({ ...V, [key]: (V[key] ? `${V[key]}\n[NOTE] ${note}` : `[NOTE] ${note}`) }));
    }
    setRows((R) => R.map((x) => (x.idx === detailRow.idx ? { ...x, status: "approved" } : x)));
    setDetailOpen(false);
  };
  const retry = async () => {
    if (!detailRow) return;
    const i = rows.findIndex((x) => x.idx === detailRow.idx);
    setDetailOpen(false);
    await runOne(i);
  };
  const sendBack = async (toTaskId: number) => {
    setDetailOpen(false);
    const i = rows.findIndex((x) => x.idx === toTaskId);
    setCursor(Math.max(0, i));
    await runAll(Math.max(0, i));
  };

  function openRowDetail(r: ExecTaskRow) {
    // Sync selection highlight to the clicked row
    setCursor(() => {
      const i = rows.findIndex((x) => x.idx === r.idx);
      return i >= 0 ? i : 0;
    });
    const e = loadEntries().find((x) => x.runId === runId && x.taskId === r.idx);
    setDetailRow(r);
    if (e) setDetailIO({ prompt: e.value.prompt, response: e.value.response, inputs: e.value.inputs, outputs: e.value.outputs });
    setDetailOpen(true);
  }

  return (
    <>
      <Card
        title={<h3 className="m-0">実行</h3>}
        right={
          <div className="flex items-center gap-2">
            <input className="input" style={{ width: 220 }} value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="run name" />
            <label className="chip-button" style={{ gap: 6 }}>
              <input type="checkbox" checked={pauseOnCheck} onChange={(e) => setPauseOnCheck(e.target.checked)} /> ✓で一時中断
            </label>
            <button className="btn btn-primary" onClick={() => runAll(0)} disabled={runStatus === "running"}>全実行</button>
            <button className="btn" onClick={() => runOne(cursor)} disabled={runStatus === "running"}>現在タスクのみ</button>
            <button className="btn" onClick={() => setCursor((c) => Math.min(c + 1, rows.length - 1))} disabled={runStatus === "running"}>カーソル→</button>
            <button className="btn" onClick={() => { stopRef.current = true; }} disabled={runStatus !== "running"}>停止</button>
            <button className="btn" onClick={resetRun} disabled={runStatus === "running"}>リセット</button>
          </div>
        }
      >
        <div style={{ overflowX: "auto" }}>
          <table className="table-base" style={{ minWidth: 1000 }}>
            <thead>
              <tr className="thead-muted">
                <th className="cell">#</th>
                <th className="cell">taskId</th>
                <th className="cell" style={{ width: 240 }}>label</th>
                <th className="cell">persona/tools</th>
                <th className="cell">in</th>
                <th className="cell">out</th>
                <th className="cell">✓</th>
                <th className="cell">status</th>
                <th className="cell">start</th>
                <th className="cell">end</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => {
                const node = tasks.find((n) => n.nodeindex === r.idx);
                const liveLabel = node ? (extractTaskBaseName(node.value) || node.value) : r.label;
                return (
                  <tr key={r.idx} style={{ background: i === cursor ? "#eef2ff" : undefined, cursor: "pointer" }} onClick={() => openRowDetail(r)}>
                    <td className="cell">{i}</td>
                    <td className="cell">{r.idx}</td>
                    <td className="cell" title={liveLabel} style={{ maxWidth: 280, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{liveLabel}</td>
                    <td className="cell">{[r.persona, ...(r.tools || [])].filter(Boolean).join(", ")}</td>
                    <td className="cell">{(r.inputs || []).join(", ")}</td>
                    <td className="cell">{(r.outputs || []).join(", ")}</td>
                    <td className="cell">{r.requiresApproval ? "✓" : "-"}</td>
                    <td className="cell">{r.status}</td>
                    <td className="cell">{r.startedAt ? new Date(r.startedAt).toLocaleTimeString() : "-"}</td>
                    <td className="cell">{r.finishedAt ? new Date(r.finishedAt).toLocaleTimeString() : "-"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: 8 }}>
          <label className="text-xs">ランタイム ($)</label>
          <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[200px] overflow-y-auto">
            {JSON.stringify(vars, null, 2)}
          </pre>
        </div>
      </Card>

      <ExecDetailModal
        open={detailOpen}
        onClose={() => setDetailOpen(false)}
        row={detailRow}
        lastIO={detailIO}
        predecessors={detailRow ? predecessorsOf(detailRow.idx) : []}
        actions={{
          approve,
          retry,
          sendBack,
          copy: () => {
            if (!detailIO) return;
            navigator.clipboard?.writeText(JSON.stringify(detailIO, null, 2));
          },
        }}
      />
    </>
  );
};

// [Spec] U-4b HistoryTab
// Purpose: 実行履歴(run/entry) の参照・フィルタ・詳細表示を行う。
// Inputs: localStorageからのラン/履歴一覧。
// Outputs: フィルタリングUI、詳細JSONプレビュー、最新状態リフレッシュ。
// Notes: ExecuteTabが積み上げたlocalStorageデータを読み取る。runFilter/taskFilter/textで絞り込み。
const HistoryTab: React.FC = () => {
  const [runs, setRuns] = React.useState<RunMeta[]>(() => loadRuns());
  const [entries, setEntries] = React.useState<HistoryEntry[]>(() => loadEntries());
  const [runFilter, setRunFilter] = React.useState<string>("");
  const [taskFilter, setTaskFilter] = React.useState<string>("");
  const [text, setText] = React.useState<string>("");

  const filtered = entries.filter((e) => (!runFilter || e.runId === runFilter) && (!taskFilter || String(e.taskId) === taskFilter) && (!text || (e.value.response || "").includes(text) || (e.value.prompt || "").includes(text)));

  const refresh = () => { setRuns(loadRuns()); setEntries(loadEntries()); };

  return (
    <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: 12 }}>
      <Card title={<h3 className="m-0">ラン一覧</h3>} right={<button className="btn" onClick={refresh}>更新</button>}>
        <div className="bg-white border border-slate-200 rounded-lg p-2 max-h-[60vh] overflow-y-auto">
          {runs.map((r) => (
            <div key={r.runId} className="border-b border-slate-100 py-1 cursor-pointer" onClick={() => setRunFilter(r.runId)}>
              <div><b>{r.name}</b> <span className="text-slate-500">({r.runId})</span></div>
              <div className="text-xs text-slate-600">{new Date(r.startedAt).toLocaleString()} {r.finishedAt ? `→ ${new Date(r.finishedAt).toLocaleString()}` : ""}</div>
            </div>
          ))}
          {!runs.length && <div style={{ color: "#6b7280" }}>まだランがありません</div>}
        </div>
      </Card>

      <Card
        title={<h3 className="m-0">履歴</h3>}
        right={
          <div className="flex items-center gap-2">
            <input className="input" style={{ width: 160 }} placeholder="runId" value={runFilter} onChange={(e) => setRunFilter(e.target.value)} />
            <input className="input" style={{ width: 80 }} placeholder="taskId" value={taskFilter} onChange={(e) => setTaskFilter(e.target.value)} />
            <input className="input" style={{ width: 180 }} placeholder="full-text" value={text} onChange={(e) => setText(e.target.value)} />
          </div>
        }
      >
        <div style={{ overflowX: "auto" }}>
          <table className="table-base" style={{ minWidth: 1000 }}>
            <thead>
              <tr className="thead-muted">
                <th className="cell">ts</th><th className="cell">runId</th><th className="cell">taskId</th><th className="cell">label</th><th className="cell">persona/tools</th><th className="cell">in</th><th className="cell">out</th><th className="cell">approved</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e) => (
                <tr key={e.id}>
                  <td className="cell">{new Date(e.createdAt).toLocaleString()}</td>
                  <td className="cell">{e.runId}</td>
                  <td className="cell">{e.taskId}</td>
                  <td className="cell">{e.value.meta.nodeLabel}</td>
                  <td className="cell">{[e.value.meta.persona, ...(e.value.meta.tools || [])].filter(Boolean).join(", ")}</td>
                  <td className="cell">{Object.keys(e.value.inputs || {}).join(", ")}</td>
                  <td className="cell">{Object.keys(e.value.outputs || {}).join(", ")}</td>
                  <td className="cell">{e.value.meta.approved ? "✔" : e.value.meta.requiresApproval ? "待" : "-"}</td>
                </tr>
              ))}
              {!filtered.length && <tr><td className="cell" colSpan={8} style={{ color: "#6b7280" }}>該当なし</td></tr>}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 8 }}>
          <label className="text-xs">詳細（先頭）</label>
          <pre className="bg-white border border-slate-200 rounded-lg p-2 text-xs whitespace-pre-wrap max-h-[260px] overflow-y-auto">
            {filtered[0] ? JSON.stringify(filtered[0], null, 2) : "(なし)"}
          </pre>
        </div>
      </Card>
    </div>
  );
};
/* ================= Inlined: KNLSwimlaneBuilder (faithful, update-button apply only) ================ */
const KNL_NODE_WIDTH = 120;
const KNL_NODE_HEIGHT = 40;
const KNL_SWIMLANE_HEIGHT = 120;
const KNL_GRID_SIZE = 20;
const KNL_CANVAS_WIDTH = 1200;

type KNLToolType = "select" | "task" | "boundary";
interface KNLPosition { x: number; y: number }
interface KNLNode { id: string; label: string; position: KNLPosition; swimlaneId: string; inputs?: string[]; outputs?: string[]; persona?: string; description?: string }
interface KNLSegment { id: string; swimlaneId: string; x: number }
interface KNLSwimlane { id: string; label: string; y: number; height: number; nodes: KNLNode[]; segments: KNLSegment[] }
interface KNLConnection { id: string; fromSwimlaneId: string; toSwimlaneId: string; type: "sequence" | "parallel" }

const knlGenId = () => Math.random().toString(36).substr(2, 9);
const knlSnap = (v: number) => Math.round(v / KNL_GRID_SIZE) * KNL_GRID_SIZE;

const knlGenerateDSL = (swimlanes: KNLSwimlane[], connections: KNLConnection[]) => {
  if (!swimlanes.length) return "";
  const dslParts: string[] = [];
  swimlanes.forEach((lane, index) => {
    if (lane.nodes.length === 0) return;
    const sorted = [...lane.nodes].sort((a, b) => a.position.x - b.position.x);
    const parts: string[] = [];
    sorted.forEach((node, i) => {
      let nodeDsl = `^${node.label}`;
      if (node.inputs?.length) nodeDsl += node.inputs.map((s) => `($${s})`).join("");
      if (node.outputs?.length) nodeDsl += node.outputs.map((s) => `=$${s}`).join("");
      if (node.persona) nodeDsl += `@${node.persona}`;
      if (node.description) nodeDsl += `"${node.description}"`;
      if (i > 0) {
        const prev = sorted[i - 1];
        const hasBoundary = lane.segments.some((seg) => seg.x > prev.position.x && seg.x < node.position.x);
        parts.push(hasBoundary ? "&" : "|");
      }
      parts.push(nodeDsl);
    });
    const laneDsl = parts.join("");
    if (laneDsl) dslParts.push(laneDsl);
    if (index < swimlanes.length - 1) {
      const conn = connections.find((c) => c.fromSwimlaneId === lane.id);
      dslParts.push(conn?.type === "parallel" ? "§" : "↓");
    }
  });
  return dslParts.join("");
};

// [Spec] U-2e KNLSwimlaneBuilder
// Purpose: GUIでスイムレーン/タスク配置→DSL生成を行う実験的ツール。
// Inputs: onApplyDslコールバック。内部でスイムレーン・接続stateを保持。
// Outputs: DSL文字列の生成/適用、トースト通知。
// Notes: 現状UIセクション内部で独立。更新ボタンでのみ親DSLに反映。
function KNLSwimlaneBuilder({
  onApplyDsl,
  parsed,
}: {
  onApplyDsl?: (dsl: string) => void;
  parsed?: KNL07ParseResult;
}) {
  const [swimlanes, setSwimlanes] = React.useState<KNLSwimlane[]>([
    { id: knlGenId(), label: "メインフロー", y: 0, height: KNL_SWIMLANE_HEIGHT, nodes: [], segments: [] },
  ]);
  const [connections, setConnections] = React.useState<KNLConnection[]>([]);
  const [selectedTool, setSelectedTool] = React.useState<KNLToolType>("task");
  const [selectedNodeId, setSelectedNodeId] = React.useState<string | null>(null);
  const [draggedNodeId, setDraggedNodeId] = React.useState<string | null>(null);
  const [dragOffset, setDragOffset] = React.useState<KNLPosition>({ x: 0, y: 0 });
  const [toastMessage, setToastMessage] = React.useState<string | null>(null);
  const svgRef = React.useRef<SVGSVGElement>(null);

  const showToast = (m: string) => { setToastMessage(m); setTimeout(() => setToastMessage(null), 3000); };
  const dsl = knlGenerateDSL(swimlanes, connections);
  const canvasHeight = Math.max(400, swimlanes.length * (KNL_SWIMLANE_HEIGHT + 10) + 100);

  // Import current parsed DSL -> one-lane rough layout
  const importFromParsed = React.useCallback(() => {
    if (!parsed) { showToast("DSLを解析できません"); return; }
    const taskNodes = (parsed.nodes.filter((n) => (n as any) instanceof (TaskToken as any)) as NodeToken[]) || [];
    if (!taskNodes.length) { showToast("インポート可能なタスクがありません"); return; }
    // トポ順で横一列に配置
    const order = (() => {
      try {
        // build topo order from edges in parsed.nodes
        const indeg = new Map<number, number>();
        const out = new Map<number, number[]>();
        taskNodes.forEach((n) => {
          if (n.nodeindex == null) return;
          indeg.set(n.nodeindex, 0);
          out.set(n.nodeindex, []);
        });
        taskNodes.forEach((n) => {
          (n.to || []).forEach((t) => {
            if (!indeg.has(t)) return;
            indeg.set(t, (indeg.get(t) || 0) + 1);
            out.get(n.nodeindex!)!.push(t);
          });
        });
        const q: number[] = [];
        indeg.forEach((d, i) => { if (d === 0) q.push(i); });
        const topo: number[] = [];
        while (q.length) {
          const u = q.shift()!;
          topo.push(u);
          (out.get(u) || []).forEach((v) => {
            const d = (indeg.get(v) || 0) - 1;
            indeg.set(v, d);
            if (d === 0) q.push(v);
          });
        }
        // include isolated
        taskNodes.forEach((n) => { const i = n.nodeindex!; if (!topo.includes(i)) topo.push(i); });
        return topo;
      } catch {
        return taskNodes.map((t) => t.nodeindex!).filter((x) => x != null);
      }
    })();
    const laneId = knlGenId();
    const baseY = 0;
    const nodes: KNLNode[] = order.map((idx, i) => {
      const t = taskNodes.find((n) => n.nodeindex === idx);
      const label = (t?.value || "").replace(/^\^\??/, "");
      const persona = ((t?.attributes as any)?.persona as string) || undefined;
      return {
        id: knlGenId(),
        label: label || `Node ${idx}`,
        position: { x: knlSnap(40 + i * (KNL_NODE_WIDTH + 40)), y: baseY + 40 },
        swimlaneId: laneId,
        inputs: (t?.explicitInputs || []).map((s) => s.replace(/^\$+/, "")),
        outputs: (t?.explicitOutputs || []).map((s) => s.replace(/^=\$+/, "")),
        persona,
        description: (t?.attributes as any)?.description || "",
      };
    });
    const newLane: KNLSwimlane = { id: laneId, label: "インポート", y: baseY, height: KNL_SWIMLANE_HEIGHT, nodes, segments: [] };
    setSwimlanes([newLane]);
    setConnections([]);
    showToast("DSLを図にインポートしました");
  }, [parsed]);

  const findNode = (nodeId: string): KNLNode | undefined => {
    for (const lane of swimlanes) {
      const node = lane.nodes.find((n) => n.id === nodeId);
      if (node) return node;
    }
    return undefined;
  };

  React.useEffect(() => {
    if (!draggedNodeId) return;
    const handleMove = (e: MouseEvent) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const x = knlSnap(e.clientX - rect.left - dragOffset.x);
      const y = e.clientY - rect.top - dragOffset.y;
      setSwimlanes((prev) => prev.map((lane) => ({
        ...lane,
        nodes: lane.nodes.map((node) => node.id === draggedNodeId ? { ...node, position: { x, y } } : node),
      })));
    };
    const handleUp = () => {
      if (!draggedNodeId) { setDraggedNodeId(null); return; }
      const node = findNode(draggedNodeId);
      if (!node) { setDraggedNodeId(null); return; }
      const targetLane = swimlanes.find((lane) => node.position.y >= lane.y && node.position.y <= lane.y + lane.height);
      if (targetLane && targetLane.id !== node.swimlaneId) {
        setSwimlanes((prev) => {
          const removed = prev.map((lane) => ({ ...lane, nodes: lane.nodes.filter((n) => n.id !== draggedNodeId) }));
          return removed.map((lane) => {
            if (lane.id === targetLane.id) {
              const moved = { ...node, swimlaneId: targetLane.id, position: { ...node.position, y: targetLane.y + 40 } };
              return { ...lane, nodes: [...lane.nodes, moved] };
            }
            return lane;
          });
        });
      } else if (targetLane) {
        setSwimlanes((prev) => prev.map((lane) => ({
          ...lane,
          nodes: lane.nodes.map((n) => (n.id === draggedNodeId ? { ...n, position: { x: n.position.x, y: lane.y + 40 } } : n)),
        })));
      }
      setDraggedNodeId(null);
    };
    document.addEventListener("mousemove", handleMove);
    document.addEventListener("mouseup", handleUp);
    return () => {
      document.removeEventListener("mousemove", handleMove);
      document.removeEventListener("mouseup", handleUp);
    };
  }, [draggedNodeId, dragOffset, swimlanes]);

  // キャンバスクリック: GUI更新のみ。DSLの反映は「更新」ボタンで行う
  const handleCanvasClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const target = e.target as SVGElement;
    if (target.getAttribute("data-interactive") === "true") return;
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (selectedTool === "task") {
      const targetLane = swimlanes.find((lane) => y >= lane.y && y <= lane.y + lane.height);
      if (!targetLane) { showToast("スイムレーン内にノードを配置してください"); return; }
      const newNode: KNLNode = { id: knlGenId(), label: "新しいタスク", position: { x: knlSnap(x), y: targetLane.y + 40 }, swimlaneId: targetLane.id };
      setSwimlanes((prev) => prev.map((lane) => (lane.id === targetLane.id ? { ...lane, nodes: [...lane.nodes, newNode] } : lane)));
      setSelectedNodeId(newNode.id);
      showToast("ノードを追加しました");
    } else if (selectedTool === "boundary") {
      const targetLane = swimlanes.find((lane) => y >= lane.y && y <= lane.y + lane.height);
      if (!targetLane) { showToast("スイムレーン内に境界を配置してください"); return; }
      const newSeg: KNLSegment = { id: knlGenId(), swimlaneId: targetLane.id, x: knlSnap(x) };
      setSwimlanes((prev) => prev.map((lane) => (lane.id === targetLane.id ? { ...lane, segments: [...lane.segments, newSeg] } : lane)));
      showToast("境界を追加しました");
    } else if (selectedTool === "select") {
      setSelectedNodeId(null);
    }
  };

  const handleNodeMouseDown = (e: React.MouseEvent, node: KNLNode) => {
    if (selectedTool !== "select") return;
    e.preventDefault();
    e.stopPropagation();
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect) return;
    setDraggedNodeId(node.id);
    setDragOffset({ x: e.clientX - rect.left - node.position.x, y: e.clientY - rect.top - node.position.y });
    setSelectedNodeId(node.id);
  };

  const addSwimlane = () => {
    const newLane: KNLSwimlane = { id: knlGenId(), label: `レーン ${swimlanes.length + 1}`, y: swimlanes.length * (KNL_SWIMLANE_HEIGHT + 10), height: KNL_SWIMLANE_HEIGHT, nodes: [], segments: [] };
    setSwimlanes([...swimlanes, newLane]);
    showToast("スイムレーンを追加しました");
  };

  const clearCanvas = () => {
    if (confirm("すべての要素をクリアしますか？")) {
      setSwimlanes([{ id: knlGenId(), label: "メインフロー", y: 0, height: KNL_SWIMLANE_HEIGHT, nodes: [], segments: [] }]);
      setConnections([]);
      setSelectedNodeId(null);
      setSelectedTool("task");
      showToast("キャンバスをクリアしました");
    }
  };

  // ここでのみ DSL を親へ反映（GUI操作では反映しない）
  const applyDsl = () => {
    if (!onApplyDsl) return;
    onApplyDsl(dsl);
    showToast("DSLを更新しました");
  };

  const selectedNode = selectedNodeId ? findNode(selectedNodeId) : null;

  return (
    <div className="h-screen bg-gray-50 flex flex-col">
      <div className="bg-white border-b p-4">
        <h1 className="text-2xl font-bold">KNL スイムレーンDSLビルダー</h1>
        <p className="text-sm text-gray-600">GUIで編集し、「更新」ボタンでDSLに反映</p>
      </div>
      <div className="bg-white border-b p-4 flex gap-4">
        <div className="flex gap-2">
          <button className={`px-3 py-1 rounded ${selectedTool === "select" ? "bg-blue-500 text-white" : "bg-gray-200"}`} onClick={() => setSelectedTool("select")}>
            選択
          </button>
          <button className={`px-3 py-1 rounded ${selectedTool === "task" ? "bg-blue-500 text-white" : "bg-gray-200"}`} onClick={() => setSelectedTool("task")}>
            タスク
          </button>
          <button className={`px-3 py-1 rounded ${selectedTool === "boundary" ? "bg-blue-500 text-white" : "bg-gray-200"}`} onClick={() => setSelectedTool("boundary")}>
            境界
          </button>
        </div>
        <div className="flex gap-2">
          <button className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300" onClick={addSwimlane}>レーン追加</button>
          <button className="px-3 py-1 rounded bg-red-100 text-red-600 hover:bg-red-200" onClick={clearCanvas}>クリア</button>
          <button className="px-3 py-1 rounded bg-gray-200 hover:bg-gray-300" onClick={importFromParsed}>インポート（DSL→図）</button>
          {onApplyDsl && (
            <button
              className="px-3 py-1 rounded bg-blue-500 text-white hover:bg-blue-600"
              onClick={applyDsl}
            >
              更新
            </button>
          )}
        </div>
      </div>
      <div className="flex-1 flex flex-col p-4 gap:4 overflow-hidden">
        <div className="flex-1 flex gap-4">
          <div className="flex-1 bg-white border rounded overflow-auto">
            <svg ref={svgRef} width={KNL_CANVAS_WIDTH} height={canvasHeight} className={selectedTool === "select" ? "cursor-default" : "cursor-crosshair"} onClick={handleCanvasClick}>
              <defs>
                <pattern id="grid" width={KNL_GRID_SIZE} height={KNL_GRID_SIZE} patternUnits="userSpaceOnUse">
                  <circle cx="1" cy="1" r="1" fill="#e5e7eb" />
                </pattern>
              </defs>
              <rect width="100%" height="100%" fill="url(#grid)" />
              {swimlanes.map((lane, index) => (
                <g key={lane.id}>
                  <rect x={0} y={lane.y} width={KNL_CANVAS_WIDTH} height={lane.height} fill="rgba(229, 231, 235, 0.3)" stroke="#94a3b8" strokeWidth={2} />
                  <text x={10} y={lane.y + 20} fontSize="14" fontWeight="500">{lane.label}</text>
                  {lane.segments.map((seg) => (
                    <g key={seg.id}>
                      <line x1={seg.x} y1={lane.y} x2={seg.x} y2={lane.y + lane.height} stroke="#f97316" strokeWidth={3} strokeDasharray="5,5" />
                      <circle cx={seg.x} cy={lane.y + 10} r={6} fill="white" stroke="#dc2626" className="cursor-pointer" data-interactive="true" onClick={() => {
                        setSwimlanes((prev) => prev.map((l) => l.id === lane.id ? { ...l, segments: l.segments.filter((s) => s.id !== seg.id) } : l));
                        showToast("境界を削除しました");
                      }} />
                    </g>
                  ))}
                  {swimlanes.length > 1 && (
                    <circle cx={1150} cy={lane.y + 20} r={8} fill="white" stroke="#dc2626" className="cursor-pointer" data-interactive="true" onClick={() => {
                      if (confirm(`${lane.label}を削除しますか？`)) {
                        setSwimlanes((prev) => prev.filter((l) => l.id !== lane.id));
                        setConnections((prev) => prev.filter((c) => c.fromSwimlaneId !== lane.id && c.toSwimlaneId !== lane.id));
                        showToast("スイムレーンを削除しました");
                      }
                    }} />
                  )}
                  {index < swimlanes.length - 1 && (
                    <g>
                      <line x1={0} y1={lane.y + lane.height + 5} x2={KNL_CANVAS_WIDTH} y2={lane.y + lane.height + 5} stroke="#6b7280" strokeWidth={2} strokeDasharray={connections.find((c) => c.fromSwimlaneId === lane.id)?.type === "parallel" ? "5,5" : "none"} />
                      <circle cx={200} cy={lane.y + lane.height + 5} r={10} fill="white" stroke="#6b7280" className="cursor-pointer" data-interactive="true" onClick={() => {
                        const conn = connections.find((c) => c.fromSwimlaneId === lane.id);
                        if (conn) {
                          setConnections((prev) => prev.map((c) => (c.id === conn.id ? { ...c, type: c.type === "sequence" ? "parallel" : "sequence" } : c)));
                        } else {
                          setConnections((prev) => [...prev, { id: knlGenId(), fromSwimlaneId: lane.id, toSwimlaneId: swimlanes[index + 1].id, type: "sequence" }]);
                        }
                      }} />
                      <text x={200} y={lane.y + lane.height + 10} fontSize="12" textAnchor="middle" pointerEvents="none">{connections.find((c) => c.fromSwimlaneId === lane.id)?.type === "parallel" ? "§" : "↓"}</text>
                    </g>
                  )}
                </g>
              ))}
              {swimlanes.flatMap((lane) => lane.nodes).map((node) => (
                <g key={node.id}>
                  <rect x={node.position.x} y={node.position.y} width={KNL_NODE_WIDTH} height={KNL_NODE_HEIGHT} rx={8} fill="#3b82f6" stroke={selectedNodeId === node.id ? "#1e40af" : "transparent"} strokeWidth={2} className="cursor-move" data-interactive="true" onMouseDown={(e) => handleNodeMouseDown(e, node)} />
                  <text x={node.position.x + KNL_NODE_WIDTH / 2} y={node.position.y + KNL_NODE_HEIGHT / 2 + 4} fontSize="12" textAnchor="middle" fill="white" pointerEvents="none">{node.label.length > 12 ? node.label.substring(0, 12) + "..." : node.label}</text>
                </g>
              ))}
            </svg>
          </div>
          {selectedNode && (
            <div className="w-80 bg-white border rounded p-4">
              <div className="flex justify-between mb-4">
                <h3 className="font-medium">プロパティ</h3>
                <button onClick={() => setSelectedNodeId(null)}>×</button>
              </div>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">タスク名</label>
                  <input type="text" className="w-full px-3 py-1 border rounded" value={selectedNode.label} onChange={(e) => {
                    const newLabel = e.target.value;
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.map((n) => (n.id === selectedNode.id ? { ...n, label: newLabel } : n)) })));
                  }} />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">入力変数</label>
                  <input type="text" className="w-full px-3 py-1 border rounded" value={selectedNode.inputs?.join(", ") || ""} onChange={(e) => {
                    const inputs = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.map((n) => (n.id === selectedNode.id ? { ...n, inputs } : n)) })));
                  }} placeholder="input1, input2" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">出力変数</label>
                  <input type="text" className="w-full px-3 py-1 border rounded" value={selectedNode.outputs?.join(", ") || ""} onChange={(e) => {
                    const outputs = e.target.value.split(",").map((s) => s.trim()).filter(Boolean);
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.map((n) => (n.id === selectedNode.id ? { ...n, outputs } : n)) })));
                  }} placeholder="output1, output2" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">ペルソナ</label>
                  <input type="text" className="w-full px-3 py-1 border rounded" value={selectedNode.persona || ""} onChange={(e) => {
                    const persona = e.target.value.trim() || undefined;
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.map((n) => (n.id === selectedNode.id ? { ...n, persona } : n)) })));
                  }} placeholder="user" />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">説明</label>
                  <textarea className="w-full px-3 py-1 border rounded" rows={3} value={selectedNode.description || ""} onChange={(e) => {
                    const description = e.target.value.trim() || undefined;
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.map((n) => (n.id === selectedNode.id ? { ...n, description } : n)) })));
                  }} placeholder="タスクの説明" />
                </div>
                <button className="w-full py-2 bg-red-500 text-white rounded hover:bg-red-600" onClick={() => {
                  if (confirm("ノードを削除しますか？")) {
                    setSwimlanes((prev) => prev.map((lane) => ({ ...lane, nodes: lane.nodes.filter((n) => n.id !== selectedNode.id) })));
                    setSelectedNodeId(null);
                    showToast("ノードを削除しました");
                  }
                }}>削除</button>
              </div>
            </div>
          )}
        </div>
        <div className="bg-white border rounded p-4">
          <div className="flex justify-between mb-2">
            <h3 className="font-medium">生成されたDSL</h3>
            <div className="flex gap-2">
              <button className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300" onClick={() => { navigator.clipboard?.writeText(dsl); showToast("DSLをコピーしました"); }}>コピー</button>
              <button className="px-3 py-1 text-sm bg-gray-200 rounded hover:bg-gray-300" onClick={() => {
                const blob = new Blob([dsl], { type: "text/plain" });
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url; a.download = "knl-workflow.dsl"; a.click();
                URL.revokeObjectURL(url);
                showToast("DSLをダウンロードしました");
              }}>ダウンロード</button>
            </div>
          </div>
          <div className="flex gap-4">
            <pre className="flex-1 p-3 bg-gray-50 rounded text-sm font-mono">{dsl || "// ノードを配置してDSLを生成"}</pre>
            {dsl && (
              <div className="text-xs text-gray-600 space-y-1">
                <div><code className="bg-gray-200 px-1">^タスク</code> タスク</div>
                <div><code className="bg-gray-200 px-1">($var)</code> 入力</div>
                <div><code className="bg-gray-200 px-1">=$var</code> 出力</div>
                <div><code className="bg-gray-200 px-1">@user</code> ペルソナ</div>
                <div><code className="bg-gray-200 px-1">"説明"</code> 説明</div>
                <div><code className="bg-gray-200 px-1">|</code> 順次</div>
                <div><code className="bg-gray-200 px-1">&</code> 並列</div>
                <div><code className="bg-gray-200 px-1">↓</code> レーン順次</div>
                <div><code className="bg-gray-200 px-1">§</code> レーン並列</div>
              </div>
            )}
          </div>
        </div>
      </div>
      {toastMessage && (
        <div className="fixed bottom-4 right-4 px-4 py-2 bg-green-500 text-white rounded shadow-lg">{toastMessage}</div>
      )}
    </div>
  );
}
/* ================================================================================================ */

/* =============== Main App =============== */
const exampleDSL = `^研究"トピック調査"@researcher=$rawData #research |
^分析"整理"($rawData)=$insights ✓
↓
^構成"アウトライン"@editor=$outline & ^参照"基礎調査"=$refs
↓
^初稿"執筆"($insights)($outline)($refs)=$content ; @memo
↓
^修正"フィードバック反映"($content)=$final ×2 > ^ガイド"方針継承"@lead`;

const DEFAULT_KNOWLEDGE: AnyKnowledge = {
  tasks: {
    research: { name: "Research", aliases: ["調査", "リサーチ"], description: "関連情報の収集と一次整理", focus: "関連情報の収集", scope: "対象領域の一次情報", constraints: ["信頼できる出典"], outputs: ["rawData"] },
    analysis: { name: "Analysis", aliases: ["分析"], description: "収集データの整理・洞察抽出", focus: "洞察抽出", scope: "収集データ", constraints: ["根拠の明示"], inputs: ["rawData"], outputs: ["insights"] },
    outline: { name: "Outline", aliases: ["構成"], description: "アウトライン作成", outputs: ["outline"] },
    draft: { name: "Draft", aliases: ["初稿"], description: "草稿の執筆", inputs: ["insights", "outline"], outputs: ["content"] },
    review: { name: "Review", aliases: ["レビュー"], description: "品質チェック", outputs: ["fb"] },
    revise: { name: "Revise", aliases: ["修正"], description: "フィードバック反映", inputs: ["content", "fb"], outputs: ["final"] },
    guide: { name: "Guide", aliases: ["ガイド"], description: "方針記述/継承", constraints: ["統一方針に準拠"] },
  },
  personas: {
    researcher: { displayName: "研究者", style: "客観・厳密", policy: "根拠重視・引用明示" },
    editor: { displayName: "編集者", style: "明快・構造化", policy: "簡潔・一貫性" },
    lead: { displayName: "リード", style: "指針・要点", policy: "一貫した方針" },
  },
  vars: {
    rawdata: { name: "rawData", description: "収集した一次情報", type: "array" },
    insights: { name: "insights", description: "分析から得た洞察", type: "text" },
    outline: { name: "outline", description: "記事アウトライン", type: "text" },
    content: { name: "content", description: "初稿本文", type: "text" },
    fb: { name: "fb", description: "レビュー指摘", type: "text" },
    final: { name: "final", description: "修正後の完成版", type: "text" },
  },
  tools: {
    memo:     { type: "memo",     motif: "メモ",           in: ["notes[]"], work: "要点をまとめる",       out: ["summary"],   aliases: ["note","scratch"] },
    library:  { type: "library",  motif: "本(ライブラリ)", in: ["refs[]"],  work: "参照を収集/要約",     out: ["briefs[]"] },
    furnace:  { type: "furnace",  motif: "炉",             in: ["raw[]"],   work: "素材を精錬する",       out: ["refined"] },
    scale:    { type: "scale",    motif: "天秤",           in: ["a","b"],   work: "比較/評価",           out: ["verdict"] },
    sign:     { type: "sign",     motif: "サイン",         in: ["facts[]"], work: "特徴抽出/記号化",     out: ["symbols[]"] },
    knife:    { type: "knife",    motif: "ナイフ",         in: ["text"],    work: "分割/抽出/切除",       out: ["chunks[]"] },
  },
};

// [Spec] U-0 Main Shell `KNL_HeaderVisual_Skeleton`
// Purpose: UI単一ファイルのroot。Reducerで全タブのStateを保持し、Tabs/Modalsを切替描画。
// Inputs: 内部初期state(exampleDSL, knowledge)、Reducer actions。
// Outputs: DSL解析結果(tokens/parsed)、各タブレンダリング、LLM設定モーダル制御。
// Notes: KNL07Coreをmemo化し、State変化で再Render。将来的な外部統合時もここをexport defaultで利用。
export default function KNL_HeaderVisual_Skeleton() {
  const [state, dispatch] = React.useReducer(reducer, {
    dsl: exampleDSL,
    attrPolicy: "lastOnly",
    mainTab: "visualize",
    selectedNodeIdx: null,
    hoveredNodeIdx: null,
    dslUndoStack: [],
    dslRedoStack: [],
    globalWish: "アウトライン付き記事の作成",
    backbone: "簡潔・再現可能・引用重視",
    knowledge: DEFAULT_KNOWLEDGE,
    history: { dslSnapshots: [] },
    settingsOpen: false,
  } as State);

  const core = React.useMemo(() => KNL07Core, []);
  const { tokens, parsed } = React.useMemo<{ tokens: KNLToken[]; parsed: KNL07ParseResult }>(() => {
    const t = core.tokenize(state.dsl);
    const p = core.parse(state.dsl, state.attrPolicy).parsed;
    const e = core.enrich(p, state.knowledge as KNL07Knowledge);
    return { tokens: t, parsed: e };
  }, [core, state.dsl, state.attrPolicy, state.knowledge]);

  const inheritance = React.useMemo(
    () => (parsed ? core.buildInheritanceIndex(parsed.executionTree) : ({} as KNL07InheritanceIndex)),
    [core, parsed],
  );

  // Node editor modal state
  const [selectedNode, setSelectedNode] = React.useState<number | null>(null);
  // Visualize toggle
  const [vizMode, setVizMode] = React.useState<"tree" | "workflow">("workflow");

  // ADD: DAG/ledger datasets
  const dagRows = React.useMemo(() => getDagNodes(parsed), [parsed]);
  const portLedger = React.useMemo(() => buildGlobalPortLedger(parsed), [parsed]);
  const flowLedger = React.useMemo(() => buildFlowLedger(parsed), [parsed]);

  return (
    <div style={{ width: "100%", margin: "0 auto", padding: 12 }}>
      <h2 style={{ margin: "8px 0 12px" }}>KNL UI v7（Topo追従 + ✓ゲート + 実行詳細）</h2>

      <HeaderCommon
        state={state}
        dispatch={dispatch}
        tokens={tokens}
        parsed={parsed}
        onOpenNode={(idx) => setSelectedNode(idx)}
      />

      <Card
        title={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(["visualize", "execute", "knowledge", "history", "swimlanebuilder"] as MainTab[]).map((t) => (
              <button
                key={t}
                onClick={() => dispatch(setMainTabAction(t))}
                className="chip-button"
                style={{ background: state.mainTab === t ? "#2563eb" : "#EDE9FE", color: state.mainTab === t ? "#fff" : "#1f2937" }}
              >
                {t}
              </button>
            ))}
          </div>
        }
      >
        {/* Visualize */}
        {parsed && state.mainTab === "visualize" && (
          <>
            <div style={{ marginBottom: 8, display: "flex", gap: 8 }}>
              <button
                className="chip-button"
                style={{ background: vizMode === "tree" ? "#2563eb" : "#fff", color: vizMode === "tree" ? "#fff" : "#1f2937", fontWeight: 700 }}
                onClick={() => setVizMode("tree")}
              >
                Tree
              </button>
              <button
                className="chip-button"
                style={{ background: vizMode === "workflow" ? "#2563eb" : "#fff", color: vizMode === "workflow" ? "#fff" : "#1f2937", fontWeight: 700 }}
                onClick={() => setVizMode("workflow")}
              >
                Workflow (SVG)
              </button>
            </div>

            {vizMode === "tree" ? (
              <Card title={<h3 className="m-0">Execution Tree</h3>}>
                <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                  {parsed.rows.map((row, i) => (
                    <div key={i} style={{ display: "flex", gap: 8 }}>
                      {row.map((item: any) => (
                        <ExecutionBlockView
                          key={item.id}
                          item={item}
                          onNodeDoubleClick={(idx) => idx != null && setSelectedNode(idx)}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </Card>
            ) : (
              <Card title={<h3 className="m-0">Workflow (SVG)</h3>}>
                <WorkflowSvg parsed={parsed} onOpenNode={(idx) => setSelectedNode(idx)} />
              </Card>
            )}

            {/* ADD: DAG ノード一覧 */}
            <Card title={<h3 className="m-0">DAG ノード一覧</h3>}>
              <div style={{ overflowX: "auto" }}>
                <table className="table-base" style={{ minWidth: 960 }}>
                  <thead>
                    <tr className="thead-muted">
                      <th className="cell">idx</th>
                      <th className="cell">value</th>
                      <th className="cell">desc</th>
                      <th className="cell">persona</th>
                      <th className="cell">binding</th>
                      <th className="cell">from</th>
                      <th className="cell">to</th>
                      <th className="cell">explicitInputs</th>
                      <th className="cell">explicitOutputs</th>
                      <th className="cell">calculatedInputs</th>
                    </tr>
                  </thead>
                  <tbody>
                    {dagRows.map((n) => (
                      <tr key={n.idx}>
                        <td className="cell">{n.idx}</td>
                        <td className="cell">{n.value}</td>
                        <td className="cell">{n.desc}</td>
                        <td className="cell">{n.persona}</td>
                        <td className="cell">{n.binding}</td>
                        <td className="cell">{(n.from || []).join(", ")}</td>
                        <td className="cell">{(n.to || []).join(", ")}</td>
                        <td className="cell">{(n.explicitInputs || []).join(", ")}</td>
                        <td className="cell">{(n.explicitOutputs || []).join(", ")}</td>
                        <td className="cell">
                          {(n.calculatedInputs || [])
                            .map((i) => `${i.input}${i.fromNode !== -1 ? ` (from ${i.fromNode})` : ""}`)
                            .join(", ")}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>

            {/* ADD: レジャー（グローバルポート／SQLライク） */}
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(420px, 1fr))",
                gap: 16,
              }}
            >
              <Card title={<h3 className="m-0">グローバルポートレジャー</h3>}>
                {portLedger.length ? (
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {portLedger.map((entry) => (
                      <div
                        key={entry.variable}
                        style={{
                          display: "grid",
                          gridTemplateColumns: "160px 1fr",
                          gap: 12,
                          fontSize: 12,
                        }}
                      >
                        <div
                          style={{
                            fontFamily:
                              'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                            color: "#b45309",
                          }}
                        >
                          {entry.variable.startsWith("$") ? entry.variable : `$${entry.variable}`}
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 4, color: "#374151" }}>
                          <div>
                            生成:{" "}
                            {entry.producers.length
                              ? entry.producers.map((p) => p.label).join(", ")
                              : "検出されませんでした"}
                          </div>
                          <div>
                            流入:{" "}
                            {entry.consumers.length
                              ? entry.consumers.map((c) => c.label).join(", ")
                              : "検出されませんでした"}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    グローバルポートは見つかりませんでした。
                  </div>
                )}
              </Card>

              <Card title={<h3 className="m-0">SQLライクレジャー</h3>}>
                {flowLedger.length ? (
                  <pre
                    style={{
                      margin: 0,
                      background: "#f9fafb",
                      borderRadius: 8,
                      padding: "12px 16px",
                      fontFamily:
                        'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                      fontSize: 12,
                      color: "#111827",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {flowLedger.join("\n")}
                  </pre>
                ) : (
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    レジャーに表示するデータがありません。
                  </div>
                )}
              </Card>
            </div>
          </>
        )}
        {state.mainTab === "swimlanebuilder" && (
          <KNLSwimlaneBuilder
            parsed={parsed}
            onApplyDsl={(dsl) =>
              dispatch(
                setDslAction(dsl, { snapshot: true, note: "Swimlane Builder" })
              )
            }
          />
        )}
        {/* Execute (keep mounted to persist jobs across tab switches) */}
        <div style={{ display: state.mainTab === "execute" ? "block" : "none" }}>
          <ExecuteTab state={state} parsed={parsed} />
        </div>

        {/* Knowledge */}
        {state.mainTab === "knowledge" && (
          <KnowledgeTab
            state={state}
            dispatch={dispatch}
            parsed={parsed}
            onOpenSettings={() => dispatch(openSettingsAction(true))}
          />
        )}

        {/* History */}
        {state.mainTab === "history" && <HistoryTab />}


      </Card>

      {/* Node Editor Modal */}
      {selectedNode != null && parsed && (
        <NodeDetailCard
          node={parsed.nodes.find((n) => n.nodeindex === selectedNode)!}
          dsl={state.dsl}
          tokens={tokens}
          onEditDsl={(dsl) => dispatch(setDslAction(dsl, { snapshot: true, note: "NodeEditor" }))}
          onClose={() => setSelectedNode(null)}
          getPrompt={() =>
            KNL07Core.buildPrompt({
              node: parsed.nodes.find((n) => n.nodeindex === selectedNode)!,
              allNodes: parsed.nodes,
              ctx: {},
              inheritance,
              global: state.globalWish,
              backbone: state.backbone,
              position: (parsed.nodePositions || {})[selectedNode],
            })
          }
          personaSystem={KNL07Core.buildPersonaInstruction(
            (parsed.nodes.find((n) => n.nodeindex === selectedNode) as any)?.attributes || {},
          )}
          inheritance={inheritance}
          nodePositions={parsed?.nodePositions || {}}
          dictTaskKey={(() => {
            const n = parsed!.nodes.find((nn) => nn.nodeindex === selectedNode) as NodeToken;
            const bindKey = (n.attributes as any)?.binding?.key;
            const base = n.value.replace(/^\^\??/, "").replace(/["@\(=].*$/, "").trim().toLowerCase();
            return bindKey || base;
          })()}
          dictTaskValue={(() => {
            const key = (() => {
              const n = parsed!.nodes.find((nn) => nn.nodeindex === selectedNode) as NodeToken;
              const bindKey = (n.attributes as any)?.binding?.key;
              const base = n.value.replace(/^\^\??/, "").replace(/["@\(=].*$/, "").trim().toLowerCase();
              return bindKey || base;
            })();
            return (state.knowledge?.tasks as any)?.[key];
          })()}
          onSaveDictTask={(k, v) => {
            const next: AnyKnowledge = JSON.parse(JSON.stringify(state.knowledge || {}));
            if (!next.tasks) next.tasks = {};
            (next.tasks as any)[k] = { ...(next.tasks as any)[k] || {}, ...v };
            dispatch(applyKnowledgeAction(next));
          }}
        />
      )}

      {/* 共通 LLM 設定モーダル */}
      <LLMSettingsModal open={state.settingsOpen} onClose={() => dispatch(openSettingsAction(false))} />
    </div>
  );
}

/* ======== Minimal styles helper (optional) ======== */
/* Example (in your CSS):
.modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;z-index:1000}
.modal-panel{background:#fff;border-radius:12px;padding:20px;min-width:560px;max-width:95vw;max-height:95vh;overflow:auto;position:relative}
.btn{border:1px solid #ddd;background:#f5f5f5;border-radius:6px;padding:6px 10px}
.btn-primary{background:#2563eb;color:#fff;border-color:#2563eb}
.btn-icon{border:1px solid #ddd;border-radius:50%;width:28px;height:28px}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;margin-bottom:12px}
.card-header{padding:10px 12px;border-bottom:1px solid #e5e7eb}
.card-content{padding:12px}
.select{border:1px solid #ddd;border-radius:6px;padding:4px 6px}
.textarea{border:1px solid #ddd;border-radius:6px;width:100%;min-height:80px;padding:6px}
.input{border:1px solid #ddd;border-radius:6px;width:100%;padding:6px}
.chip-button{border:1px solid #ddd;border-radius:999px;padding:4px 10px;background:#fff}
.table-base{width:100%;border-collapse:collapse;font-size:12px}
.table-base .cell{padding:6px;border-bottom:1px solid #e5e7eb;text-align:left;white-space:nowrap}
.thead-muted th{background:#f3f4f6}
*/