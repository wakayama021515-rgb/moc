# MOC - Collection of AI-Powered Innovation Tools

> "これ適当に作ったモックというかアイディア群" - A collection of innovative AI-powered tools and ideas

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![AI: Gemini](https://img.shields.io/badge/AI-Gemini-blue.svg)](https://ai.google.dev/)

## 📚 目次 / Table of Contents

- [概要](#概要--overview)
- [プロジェクト一覧](#プロジェクト一覧--projects)
- [評価結果](#評価結果--evaluation)
- [セットアップ](#セットアップ--setup)
- [使用方法](#使用方法--usage)
- [ライセンス](#ライセンス--license)

## 🎯 概要 / Overview

このレポジトリには、AI/ML技術を活用した革新的なアイディアとプロトタイプが含まれています。各プロジェクトは独立して動作し、実用的な問題解決にフォーカスしています。

**主な特徴**:
- 🤖 最新のLLM (Gemini) を活用
- 🎨 モダンなReact + TypeScript実装
- 🔄 多層AIエージェントアーキテクチャ
- 📊 知識グラフとベクトル検索の統合

## 📦 プロジェクト一覧 / Projects

### 1. 🌟 Node-based Chatbot (ノードチャットボット)

**ファイル**: `KNL07.tsx`, `ノードチャットボット.txt`

会話中の概念をノード（知識グラフ）として管理する高度なAIチャットボット。

**主な機能**:
- ✅ 16種類のノードタイプによる構造化知識管理
- ✅ AI-A/B/C トリプルエージェント処理
- ✅ LLMリランキングによる高精度な文脈検索
- ✅ $session.theme, $user.wish 等のメモシステム
- ✅ リアルタイムグラフ更新とエクスポート

**技術スタック**:
- React + TypeScript
- Gemini API
- Lucide Icons
- Tailwind CSS

**評価**: ⭐⭐⭐⭐⭐ (5/5) - 最も完成度が高いプロジェクト

**デモ画像**: `デモ.png`, `KNL.png`

---

### 2. 🎯 Service Concierge (社内サービスコンシェルジュ)

**ファイル**: `service-concierge.tsx.txt`

AIがユーザーの状況から最適な社内サービスを推薦するインテリジェントコンシェルジュ。

**主な機能**:
- ✅ 100個の社内サービスマスタデータ
- ✅ 動的な観点（perspective）生成
- ✅ 確率的スコアリング (0.0-1.0)
- ✅ 美しいUIと直感的なUX
- ✅ 重要度の可視化

**適用分野**:
- 企業の社内ポータル
- 新入社員オンボーディング
- 行政サービス案内
- 大学の学生支援システム

**評価**: ⭐⭐⭐⭐⭐ (4.75/5) - 即座にデプロイ可能

---

### 3. 🧠 AI Brainstorming System

**ファイル**: `ai-system-gemini-2025-09-29 (3).json`, `ブレストマン.txt`

多層AIエージェントによる体系的アイデア生成システム。

**アーキテクチャ**:
```
Layer 1: エンティティ生成 (9つの観点)
    ↓
Layer 2: SCAMPER変換 (7つの手法 × 複数回反復)
    ↓
Layer 3: 構造化分析 (グラフ、クラスタリング)
    ↓
Layer 4: 収束・統合 (シナジー評価)
    ↓
Layer 5: 最終評価 (実現可能性、革新性、ユーザー価値)
```

**SCAMPER技法**:
- S: Substitute (代替)
- C: Combine (結合)
- A: Adapt (適応)
- M: Modify (修正)
- P: Put to other use (転用)
- E: Eliminate (削除)
- R: Reverse (逆転)

**評価**: ⭐⭐⭐⭐ (4/5) - 理論的に優れているがUI実装が必要

---

### 4. 🌳 Conversation Tree Explorer

**ファイル**: `conversation-tree-explorer (2).tsx`

会話の分岐を木構造で管理・探索するシステム。

**評価**: ⭐⭐⭐⭐ (3.75/5)

---

## 📊 評価結果 / Evaluation

詳細な評価は [EVALUATION.md](EVALUATION.md) を参照してください。

### スコアサマリー

| プロジェクト | 革新性 | 実現可能性 | ユーザー価値 | 技術完成度 | 総合 |
|------------|--------|-----------|------------|----------|------|
| AI Brainstorming | 5/5 | 4/5 | 4/5 | 3/5 | **4.0/5** |
| Node Chatbot | 5/5 | 5/5 | 5/5 | 5/5 | **5.0/5** ⭐ |
| Service Concierge | 4/5 | 5/5 | 5/5 | 5/5 | **4.75/5** |
| Conversation Tree | 4/5 | 4/5 | 4/5 | 3/5 | **3.75/5** |
| ブレストマン | 5/5 | 4/5 | 4/5 | 4/5 | **4.25/5** |

**総合平均**: **4.35/5** - 優秀 (Excellent) 🎉

---

## 🚀 セットアップ / Setup

### 前提条件

- Node.js 18+ 
- npm または yarn
- Gemini API キー

### インストール手順

1. レポジトリをクローン
```bash
git clone https://github.com/wakayama021515-rgb/moc.git
cd moc
```

2. APIキーの設定
```bash
# .env ファイルを作成
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

3. プロジェクトごとのセットアップ

#### Node-based Chatbot
```bash
# React + Vite プロジェクトとして実行
npm create vite@latest chatbot -- --template react-ts
cd chatbot
npm install
# KNL07.tsx をsrc/App.tsxにコピー
npm run dev
```

#### Service Concierge
```bash
# React + Vite プロジェクトとして実行
npm create vite@latest concierge -- --template react-ts
cd concierge
npm install
# service-concierge.tsx.txt をsrc/App.tsxにコピー
npm run dev
```

---

## 💡 使用方法 / Usage

### Node-based Chatbot

1. アプリケーションを起動
2. テキスト入力欄にメッセージを入力
3. AIが重要な概念をノードとして自動抽出
4. ノードをクリックして詳細を確認
5. 「ノード」タブでグラフ全体を可視化

**ノードタグの例**:
```
<アリス#ag-alice>は<AI倫理#con-ai-ethics>について<東京#loc-tokyo>で研究しています。
```

### Service Concierge

1. 状況を自然言語で入力
   - 例: "来月から新メンバーが2人入る"
   - 例: "出張に行く"
   - 例: "データ分析したい"
2. AIが複数の観点を生成
3. 各観点で関連サービスをスコアリング
4. 推奨されたサービスを確認

### AI Brainstorming System

```bash
# JSONファイルを読み込み
cat ai-system-gemini-2025-09-29\ \(3\).json | jq '.entities | .[0:5]'

# 生成されたアイデアを確認
cat ai-system-gemini-2025-09-29\ \(3\).json | jq '.ideas'
```

---

## 🛠️ 技術スタック / Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS
- **AI/ML**: Gemini API (Google)
- **Icons**: Lucide React
- **Data**: JSON, Knowledge Graphs
- **Architecture**: Multi-agent AI Systems

---

## 📈 ロードマップ / Roadmap

### 短期（1-3ヶ月）
- [ ] README とドキュメントの充実
- [ ] GitHub Pages でのデモ公開
- [ ] テストカバレッジの追加
- [ ] セキュリティ強化（APIキー管理）

### 中期（3-6ヶ月）
- [ ] Service Concierge のSaaS化
- [ ] Node Chatbot のオープンソース化
- [ ] ビジュアライゼーション機能の強化
- [ ] マルチユーザー対応

### 長期（6-12ヶ月）
- [ ] 統合プラットフォームの構築
- [ ] エンタープライズ機能の追加
- [ ] API製品としての提供
- [ ] コミュニティ構築

---

## 🤝 コントリビューション / Contributing

現在、このプロジェクトは評価フェーズにあります。
フィードバックやアイディアは Issue にてお願いします。

---

## 📝 ライセンス / License

[MIT License](LICENSE)

---

## 🙏 謝辞 / Acknowledgments

- Google Gemini API
- React コミュニティ
- TypeScript チーム
- すべてのオープンソース貢献者

---

## 📧 コンタクト / Contact

質問やフィードバックは [GitHub Issues](https://github.com/wakayama021515-rgb/moc/issues) にてお願いします。

---

**作成日**: 2025-10-30  
**評価者**: GitHub Copilot  
**ステータス**: 評価完了 ✅
