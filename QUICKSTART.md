# クイックスタートガイド / Quick Start Guide

このガイドでは、各プロジェクトを最短で動かす方法を説明します。

## 📋 前提条件 / Prerequisites

- **Node.js**: v18以上
- **npm** または **yarn**
- **Gemini APIキー**: [こちら](https://ai.google.dev/)から取得

---

## 🚀 プロジェクト別セットアップ

### 1️⃣ Node-based Chatbot

最も完成度の高いプロジェクト。知識グラフ統合チャットボット。

#### ステップ1: プロジェクト作成
```bash
# Vite + React + TypeScript プロジェクトを作成
npm create vite@latest node-chatbot -- --template react-ts
cd node-chatbot
npm install
```

#### ステップ2: 依存関係のインストール
```bash
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

#### ステップ3: Tailwind CSS の設定

**tailwind.config.js**:
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**src/index.css**:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

#### ステップ4: コンポーネントのコピー
```bash
# KNL07.tsx を src/App.tsx にコピー
cp ../KNL07.tsx src/App.tsx
```

または、ファイルの内容を手動でコピー。

#### ステップ5: 環境変数の設定

**.env**:
```
VITE_GEMINI_API_KEY=your_api_key_here
```

**.env.example** (Git用):
```
VITE_GEMINI_API_KEY=your_gemini_api_key
```

#### ステップ6: コードの修正

**src/App.tsx** の146行目あたりを修正:
```typescript
// 修正前
const apiKey = ""; // Canvas environment provides the key automatically

// 修正後
const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
```

#### ステップ7: 起動
```bash
npm run dev
```

ブラウザで `http://localhost:5173` を開く。

#### 使い方
1. テキスト入力欄にメッセージを入力
2. 送信すると、AIが応答し、重要な概念をノードとして抽出
3. ノードをクリックして詳細を表示
4. 「ノード」タブでグラフ全体を確認

---

### 2️⃣ Service Concierge

社内サービス推薦システム。

#### ステップ1: プロジェクト作成
```bash
npm create vite@latest service-concierge -- --template react-ts
cd service-concierge
npm install
```

#### ステップ2: 依存関係のインストール
```bash
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

#### ステップ3: Tailwind CSS の設定
（Node-based Chatbot と同じ）

#### ステップ4: コンポーネントのコピー
```bash
# service-concierge.tsx.txt を src/App.tsx にコピー
cp ../service-concierge.tsx.txt src/App.tsx
```

#### ステップ5: 環境変数の設定

**.env**:
```
VITE_ANTHROPIC_API_KEY=your_claude_api_key
# または
VITE_GEMINI_API_KEY=your_gemini_api_key
```

#### ステップ6: APIの修正（オプション）

もしGeminiに変更する場合:

**src/App.tsx** の162行目あたり:
```typescript
// Anthropic (Claude) から Gemini に変更
const response = await fetch("https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=" + apiKey, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    contents: [{ parts: [{ text: prompt }] }]
  })
});
```

#### ステップ7: 起動
```bash
npm run dev
```

#### 使い方
1. 状況を自然言語で入力（例: "来月から新メンバーが2人入る"）
2. AIが複数の観点を生成
3. 各観点で関連サービスをスコアリング
4. サービスをクリックして詳細を確認

---

### 3️⃣ AI Brainstorming System (ブレストマン)

多層AIによるアイデア生成。

#### ステップ1: プロジェクト作成
```bash
npm create vite@latest brainstorm -- --template react-ts
cd brainstorm
npm install
```

#### ステップ2: 依存関係のインストール
```bash
npm install lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

#### ステップ3: Tailwind CSS の設定
（同上）

#### ステップ4: コンポーネントのコピー
```bash
# ブレストマン.txt から必要な部分を src/App.tsx にコピー
cp ../ブレストマン.txt src/App.tsx
```

#### ステップ5: 環境変数の設定

**.env**:
```
VITE_GEMINI_API_KEY=your_api_key_here
```

#### ステップ6: 起動
```bash
npm run dev
```

---

## 🔧 トラブルシューティング / Troubleshooting

### エラー: "API key not found"

**原因**: 環境変数が正しく設定されていない

**解決策**:
1. `.env` ファイルが正しい場所にあるか確認
2. ファイル名が `.env` であることを確認（`.env.txt` ではない）
3. 開発サーバーを再起動: `npm run dev`

### エラー: "Failed to fetch"

**原因**: APIキーが無効、またはCORSエラー

**解決策**:
1. APIキーが正しいか確認
2. APIキーに権限があるか確認
3. ブラウザのコンソールでエラーを確認

### エラー: "Cannot find module 'lucide-react'"

**原因**: 依存関係がインストールされていない

**解決策**:
```bash
npm install lucide-react
```

### エラー: Tailwind CSSが動作しない

**原因**: 設定が不完全

**解決策**:
1. `tailwind.config.js` の content 設定を確認
2. `src/index.css` に Tailwind directives があるか確認
3. 開発サーバーを再起動

---

## 📝 よくある質問 / FAQ

### Q1: APIキーの取得方法は？

**Gemini API**:
1. [Google AI Studio](https://ai.google.dev/) にアクセス
2. "Get API Key" をクリック
3. 無料のAPIキーを取得

**Claude API** (Service Concierge用):
1. [Anthropic Console](https://console.anthropic.com/) にアクセス
2. アカウントを作成
3. APIキーを生成

### Q2: APIコストは？

**Gemini**:
- 無料枠: 15 RPM (Requests per minute)
- 有料: 使用量に応じて課金

**Claude**:
- 無料枠: $5クレジット
- 有料: トークンベース課金

### Q3: デプロイ方法は？

**Vercel** (推奨):
```bash
npm install -g vercel
vercel
```

**Netlify**:
```bash
npm install -g netlify-cli
netlify deploy
```

### Q4: カスタマイズ方法は？

各プロジェクトのコードは自由に編集可能です。以下のファイルを確認:
- **UI**: `src/App.tsx` のJSX部分
- **ロジック**: APIコール部分
- **スタイル**: Tailwind CSSクラス

### Q5: 商用利用は可能？

はい。MIT Licenseのため商用利用可能です。ただし:
- APIの利用規約を確認してください
- APIコストを考慮してください

---

## 🎯 次のステップ

1. **基本動作の確認**
   - [ ] 各プロジェクトを起動
   - [ ] 基本機能を試す
   - [ ] エラーがないか確認

2. **カスタマイズ**
   - [ ] UIをカスタマイズ
   - [ ] 独自機能を追加
   - [ ] データを追加

3. **デプロイ**
   - [ ] Vercel/Netlifyにデプロイ
   - [ ] 環境変数を設定
   - [ ] ドメインを設定

4. **共有**
   - [ ] GitHubにプッシュ
   - [ ] README を書く
   - [ ] フィードバックを受け取る

---

## 📚 参考資料

- [EVALUATION.md](./EVALUATION.md) - 詳細な評価
- [RECOMMENDATIONS.md](./RECOMMENDATIONS.md) - 改善提案
- [README.md](./README.md) - プロジェクト概要

---

## 💬 サポート

質問やサポートが必要な場合:
1. [GitHub Issues](https://github.com/wakayama021515-rgb/moc/issues)
2. コードのコメントを確認
3. 公式ドキュメントを参照

---

**作成日**: 2025-10-30  
**更新日**: 2025-10-30  
**バージョン**: 1.0
