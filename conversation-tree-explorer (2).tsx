import React, { useState, useRef, useEffect } from 'react';
import { Settings, Zap, Lock, Unlock, GitBranch, ChevronDown, ChevronUp, Loader2 } from 'lucide-react';

// Types
type NodeId = string;
type NodeType = 'user' | 'ai' | 'goal';

type ImpulsePattern = {
  semantic: string[];
  keywords: string[];
  emotional?: string;
};

type IntentPattern = {
  goal: string;
  strategy: string;
  tone?: string;
};

type ConvNode = {
  id: NodeId;
  turn: number;
  type: NodeType;
  text: string;
  utterance: string;
  
  impulse?: ImpulsePattern;
  intent?: IntentPattern;
  
  confidence: number;
  locked?: boolean;
  sourceSubId?: string; // Which sub input created this
  x?: number;
  y?: number;
};

type Edge = { from: NodeId; to: NodeId; type?: 'forward' | 'jump' | 'back' };

type Transaction = {
  op: 'add_node' | 'update_node' | 'delete_node' | 'boost_confidence' | 'prune_branch';
  node?: ConvNode;
  nodeId?: NodeId;
  parents?: NodeId[];
  patch?: Partial<ConvNode>;
  targetPattern?: string;
};

// Graph System
class ConvGraph {
  nodes = new Map<NodeId, ConvNode>();
  edges: Edge[] = [];
  
  addNode(n: ConvNode) { this.nodes.set(n.id, n); }
  getNode(id: NodeId) { return this.nodes.get(id); }
  getAllNodes() { return Array.from(this.nodes.values()); }
  addEdge(e: Edge) { this.edges.push(e); }
  
  getChildren(id: NodeId) {
    return this.edges.filter(e => e.from === id).map(e => e.to);
  }
  
  getParents(id: NodeId) {
    return this.edges.filter(e => e.to === id).map(e => e.from);
  }
  
  deleteNode(id: NodeId) {
    this.nodes.delete(id);
    this.edges = this.edges.filter(e => e.from !== id && e.to !== id);
  }
  
  deleteSubtree(rootId: NodeId) {
    const toDel = new Set<NodeId>();
    const dfs = (id: NodeId) => {
      toDel.add(id);
      this.getChildren(id).forEach(ch => dfs(ch));
    };
    dfs(rootId);
    toDel.forEach(id => this.deleteNode(id));
  }
  
  clear() {
    this.nodes.clear();
    this.edges = [];
  }
}

// Layout
const layoutTree = (graph: ConvGraph, width: number, height: number) => {
  const nodes = graph.getAllNodes();
  const maxTurn = Math.max(...nodes.map(n => n.turn), 0);
  
  const turnWidth = width / (maxTurn + 2);
  
  for (let t = 0; t <= maxTurn; t++) {
    const turnNodes = nodes.filter(n => n.turn === t);
    const spacing = height / (turnNodes.length + 1);
    
    turnNodes.forEach((n, idx) => {
      n.x = turnWidth * (t + 1);
      n.y = spacing * (idx + 1);
    });
  }
};

// LLM Call - Full Generation
async function generateTree(mainInput: string, subInputs: string[], config: any) {
  const context = [mainInput, ...subInputs.filter(s => s.trim())].join('\n');
  
  const prompt = `Generate a conversation tree exploring future possibilities.
Context: "${context}"

Return JSON only:
{
  "nodes": [
    {
      "id": "unique_id",
      "turn": 0-${config.maxTurns},
      "type": "user|ai|goal",
      "text": "short label (15 chars max)",
      "utterance": "full statement (50-120 chars)",
      "impulse": {
        "semantic": ["question", "request", etc],
        "keywords": ["key", "words"],
        "emotional": "curious|frustrated|excited"
      },
      "intent": {
        "goal": "inform|persuade|clarify|empathize",
        "strategy": "direct|exploratory|cautious",
        "tone": "formal|casual|technical"
      },
      "confidence": 0.0-1.0,
      "parents": ["parent_id1", ...]
    }
  ]
}

Rules:
- Turn 0: ONE user node (context summary)
- Turn 1-${config.maxTurns-1}: alternate AI/user, ~${config.branches} branches each
- Turn ${config.maxTurns}: ${config.goals} goal nodes
- All nodes need impulse/intent patterns
- Be creative with branches`;

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 4000,
      temperature: 1.0,
      messages: [{ role: 'user', content: prompt }]
    })
  });
  
  const data = await res.json();
  let text = data?.content?.[0]?.text || '';
  text = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
  return JSON.parse(text);
}

// LLM Call - Differential Transactions
async function generateTransactions(existingTree: any, newSubInput: string, subId: string) {
  const prompt = `You are updating an existing conversation tree with new context.

EXISTING TREE:
${JSON.stringify(existingTree, null, 2)}

NEW CONTEXT:
"${newSubInput}"

Generate transactions to integrate this new context. Return JSON only:
{
  "transactions": [
    {
      "op": "add_node|update_node|boost_confidence|prune_branch",
      "node": { ... full node for add_node },
      "nodeId": "id for update/delete/boost",
      "parents": ["parent_ids for add_node"],
      "patch": { ... partial update for update_node },
      "targetPattern": "pattern to boost/prune",
      "sourceSubId": "${subId}"
    }
  ]
}

Operations:
- add_node: Add new branches that explore this context
- update_node: Modify existing nodes to reflect new info
- boost_confidence: Increase confidence of nodes matching this theme
- prune_branch: Decrease confidence of nodes contradicting this

Do NOT modify locked nodes. Focus on enriching the tree with new perspectives.`;

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'claude-sonnet-4-20250514',
      max_tokens: 2000,
      temperature: 0.8,
      messages: [{ role: 'user', content: prompt }]
    })
  });
  
  const data = await res.json();
  let text = data?.content?.[0]?.text || '';
  text = text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
  return JSON.parse(text);
}

// Apply Transactions
function applyTransactions(graph: ConvGraph, transactions: Transaction[]) {
  for (const tx of transactions) {
    try {
      if (tx.op === 'add_node' && tx.node) {
        const node: ConvNode = {
          id: String(tx.node.id),
          turn: Number(tx.node.turn) || 0,
          type: (tx.node.type as NodeType) || 'user',
          text: String(tx.node.text || '').slice(0, 15),
          utterance: String(tx.node.utterance || ''),
          impulse: tx.node.impulse,
          intent: tx.node.intent,
          confidence: Math.max(0, Math.min(1, tx.node.confidence ?? 0.8)),
          sourceSubId: tx.node.sourceSubId,
        };
        graph.addNode(node);
        
        const parents = tx.parents || [];
        parents.forEach(p => {
          graph.addEdge({ from: String(p), to: node.id, type: 'forward' });
        });
      } else if (tx.op === 'update_node' && tx.nodeId && tx.patch) {
        const node = graph.getNode(tx.nodeId);
        if (node && !node.locked) {
          Object.assign(node, tx.patch);
        }
      } else if (tx.op === 'boost_confidence' && tx.targetPattern) {
        graph.getAllNodes().forEach(n => {
          if (!n.locked && (
            n.text.toLowerCase().includes(tx.targetPattern!.toLowerCase()) ||
            n.utterance.toLowerCase().includes(tx.targetPattern!.toLowerCase())
          )) {
            n.confidence = Math.min(1, n.confidence * 1.2);
          }
        });
      } else if (tx.op === 'prune_branch' && tx.targetPattern) {
        graph.getAllNodes().forEach(n => {
          if (!n.locked && (
            n.text.toLowerCase().includes(tx.targetPattern!.toLowerCase()) ||
            n.utterance.toLowerCase().includes(tx.targetPattern!.toLowerCase())
          )) {
            n.confidence = Math.max(0.1, n.confidence * 0.6);
          }
        });
      } else if (tx.op === 'delete_node' && tx.nodeId) {
        const node = graph.getNode(tx.nodeId);
        if (node && !node.locked) {
          graph.deleteNode(tx.nodeId);
        }
      }
    } catch (e) {
      console.warn('Transaction failed:', tx, e);
    }
  }
}

// Main Component
export default function ConversationTreeExplorer() {
  const [mainInput, setMainInput] = useState('');
  const [subInputs, setSubInputs] = useState<{ id: string; value: string }[]>([
    { id: 'sub-0', value: '' }
  ]);
  
  const [config, setConfig] = useState({
    maxTurns: 5,
    branches: 3,
    goals: 2
  });
  const [showConfig, setShowConfig] = useState(false);
  
  const graphRef = useRef(new ConvGraph());
  const [version, setVersion] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [genStatus, setGenStatus] = useState('');
  
  const [selectedId, setSelectedId] = useState<NodeId | null>(null);
  const [hoverId, setHoverId] = useState<NodeId | null>(null);
  
  const [topOpen, setTopOpen] = useState(true);
  const [bottomOpen, setBottomOpen] = useState(true);
  
  const containerRef = useRef<HTMLDivElement>(null);
  const [canvasSize, setCanvasSize] = useState({ width: 1400, height: 600 });
  
  const prevMainRef = useRef('');
  const prevSubsRef = useRef<{ id: string; value: string }[]>([]);
  const debounceTimerRef = useRef<number | null>(null);
  
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setCanvasSize({ width: rect.width - 40, height: rect.height - 40 });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [topOpen, bottomOpen]);
  
  const bump = () => setVersion(v => v + 1);
  
  // Auto-generation with debounce
  useEffect(() => {
    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }
    
    debounceTimerRef.current = window.setTimeout(() => {
      handleAutoGenerate();
    }, 1500);
    
    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [mainInput, subInputs]);
  
  const handleAutoGenerate = async () => {
    const graph = graphRef.current;
    
    // Main input changed - full regeneration (except locked)
    if (mainInput !== prevMainRef.current && mainInput.trim()) {
      setGenStatus('Regenerating main tree...');
      setIsGenerating(true);
      
      try {
        const lockedNodes = graph.getAllNodes().filter(n => n.locked);
        const data = await generateTree(mainInput, subInputs.map(s => s.value), config);
        
        // Clear unlocked nodes
        const allNodes = graph.getAllNodes();
        allNodes.forEach(n => {
          if (!n.locked) graph.deleteNode(n.id);
        });
        
        // Add new nodes
        const nodes = data.nodes || [];
        nodes.forEach((n: any) => {
          const node: ConvNode = {
            id: String(n.id),
            turn: Number(n.turn) || 0,
            type: (n.type as NodeType) || 'user',
            text: String(n.text || '').slice(0, 15),
            utterance: String(n.utterance || ''),
            impulse: n.impulse,
            intent: n.intent,
            confidence: Math.max(0, Math.min(1, n.confidence ?? 0.8)),
          };
          graph.addNode(node);
          
          const parents = Array.isArray(n.parents) ? n.parents : [];
          parents.forEach((p: any) => {
            graph.addEdge({ from: String(p), to: node.id, type: 'forward' });
          });
        });
        
        // Re-add locked nodes
        lockedNodes.forEach(n => graph.addNode(n));
        
        layoutTree(graph, canvasSize.width, canvasSize.height);
        bump();
        prevMainRef.current = mainInput;
      } catch (e) {
        console.error(e);
      } finally {
        setIsGenerating(false);
        setGenStatus('');
      }
      return;
    }
    
    // Sub inputs changed - differential update
    const prevSubMap = new Map(prevSubsRef.current.map(s => [s.id, s.value]));
    const currSubMap = new Map(subInputs.map(s => [s.id, s.value]));
    
    for (const sub of subInputs) {
      const prevValue = prevSubMap.get(sub.id) || '';
      const currValue = sub.value.trim();
      
      if (currValue && currValue !== prevValue) {
        setGenStatus(`Updating with sub input ${sub.id}...`);
        setIsGenerating(true);
        
        try {
          const existingTree = {
            nodes: graph.getAllNodes().map(n => ({
              id: n.id,
              type: n.type,
              turn: n.turn,
              text: n.text,
              utterance: n.utterance,
              confidence: n.confidence,
              locked: n.locked,
              parents: graph.getParents(n.id),
            }))
          };
          
          const data = await generateTransactions(existingTree, currValue, sub.id);
          applyTransactions(graph, data.transactions || []);
          
          layoutTree(graph, canvasSize.width, canvasSize.height);
          bump();
        } catch (e) {
          console.error(e);
        } finally {
          setIsGenerating(false);
          setGenStatus('');
        }
      }
    }
    
    // Check for deleted sub inputs
    for (const [oldId, oldValue] of prevSubMap.entries()) {
      if (!currSubMap.has(oldId) && oldValue.trim()) {
        // Remove nodes associated with this sub input
        graph.getAllNodes().forEach(n => {
          if (n.sourceSubId === oldId && !n.locked) {
            graph.deleteNode(n.id);
          }
        });
        layoutTree(graph, canvasSize.width, canvasSize.height);
        bump();
      }
    }
    
    prevSubsRef.current = [...subInputs];
  };
  
  const toggleLock = (id: NodeId) => {
    const node = graphRef.current.getNode(id);
    if (node) {
      node.locked = !node.locked;
      bump();
    }
  };
  
  const confirmNode = (id: NodeId) => {
    const node = graphRef.current.getNode(id);
    if (!node) return;
    
    const siblings = graphRef.current.getAllNodes()
      .filter(n => n.turn === node.turn && n.id !== id);
    
    siblings.forEach(s => graphRef.current.deleteSubtree(s.id));
    layoutTree(graphRef.current, canvasSize.width, canvasSize.height);
    bump();
  };
  
  const graph = graphRef.current;
  const nodes = graph.getAllNodes();
  const edges = graph.edges;
  const selected = selectedId ? graph.getNode(selectedId) : null;
  
  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white">
      {/* Top Panel */}
      {topOpen && (
        <div className="border-b border-gray-800 p-3">
          <div className="flex items-start gap-3">
            <div className="flex-1 flex gap-3">
              <div className="flex-1">
                <label className="text-xs text-gray-400 mb-1 block">Main Input (auto-regenerates on change)</label>
                <textarea
                  value={mainInput}
                  onChange={e => setMainInput(e.target.value)}
                  placeholder="What to explore?"
                  rows={2}
                  className="w-full bg-gray-800 rounded p-2 text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none"
                />
              </div>
              
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <label className="text-xs text-gray-400">Sub Inputs (adds differential updates)</label>
                  <button 
                    onClick={() => setSubInputs([...subInputs, { id: `sub-${Date.now()}`, value: '' }])}
                    className="text-xs px-1.5 py-0.5 bg-gray-800 hover:bg-gray-700 rounded"
                  >
                    +
                  </button>
                </div>
                <div className="flex gap-1">
                  {subInputs.map((sub, i) => (
                    <div key={sub.id} className="flex-1">
                      <textarea
                        value={sub.value}
                        onChange={e => {
                          const next = [...subInputs];
                          next[i] = { ...sub, value: e.target.value };
                          setSubInputs(next);
                        }}
                        placeholder={`Context ${i + 1}`}
                        rows={2}
                        className="w-full bg-gray-800 rounded p-2 text-xs focus:ring-1 focus:ring-blue-500 focus:outline-none"
                      />
                      {subInputs.length > 1 && (
                        <button
                          onClick={() => setSubInputs(subInputs.filter(s => s.id !== sub.id))}
                          className="text-xs text-gray-500 hover:text-gray-300 mt-0.5"
                        >
                          ✕
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            <div className="flex flex-col gap-2 w-32">
              {isGenerating ? (
                <div className="py-1.5 bg-blue-600/50 rounded text-xs flex items-center justify-center gap-1">
                  <Loader2 size={14} className="animate-spin" />
                  Auto
                </div>
              ) : (
                <div className="py-1.5 bg-green-900/30 text-green-400 rounded text-xs flex items-center justify-center gap-1">
                  <Zap size={14} />
                  Live
                </div>
              )}
              
              <button 
                onClick={() => setShowConfig(!showConfig)}
                className="py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-xs flex items-center justify-center gap-1"
              >
                <Settings size={14} />
                Config
              </button>
              
              <button 
                onClick={() => setTopOpen(false)}
                className="py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-xs flex items-center justify-center"
              >
                <ChevronUp size={14} />
              </button>
            </div>
          </div>
          
          {genStatus && (
            <div className="mt-2 text-xs text-blue-400 flex items-center gap-2">
              <Loader2 size={12} className="animate-spin" />
              {genStatus}
            </div>
          )}
          
          {showConfig && (
            <div className="mt-2 flex gap-4 bg-gray-800/50 p-2 rounded text-xs">
              <div className="flex-1">
                <label>Turns: {config.maxTurns}</label>
                <input 
                  type="range" 
                  min={3} 
                  max={8} 
                  value={config.maxTurns}
                  onChange={e => setConfig({...config, maxTurns: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
              <div className="flex-1">
                <label>Branches: {config.branches}</label>
                <input 
                  type="range" 
                  min={2} 
                  max={5} 
                  value={config.branches}
                  onChange={e => setConfig({...config, branches: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
              <div className="flex-1">
                <label>Goals: {config.goals}</label>
                <input 
                  type="range" 
                  min={1} 
                  max={4} 
                  value={config.goals}
                  onChange={e => setConfig({...config, goals: parseInt(e.target.value)})}
                  className="w-full"
                />
              </div>
            </div>
          )}
        </div>
      )}
      
      {!topOpen && (
        <div className="border-b border-gray-800 flex items-center justify-center h-8 hover:bg-gray-900 cursor-pointer" onClick={() => setTopOpen(true)}>
          <ChevronDown size={14} />
        </div>
      )}
      
      {/* Canvas */}
      <div ref={containerRef} className="flex-1 flex items-center justify-center p-4">
        {nodes.length === 0 ? (
          <div className="text-center text-gray-500">
            <Zap className="mx-auto mb-3 opacity-60" size={48} />
            <div className="text-lg">Type in Main Input to start</div>
            <div className="text-sm mt-2 opacity-70">Tree generates automatically after 1.5s</div>
          </div>
        ) : (
          <svg width={canvasSize.width} height={canvasSize.height} className="bg-gray-900 rounded-lg">
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="10" 
                      refX="9" refY="3" orient="auto">
                <polygon points="0 0, 10 3, 0 6" fill="#6B7280" />
              </marker>
            </defs>
            
            <g>
              {edges.map((e, i) => {
                const from = graph.getNode(e.from);
                const to = graph.getNode(e.to);
                if (!from?.x || !to?.x) return null;
                
                const isHighlighted = hoverId === e.from || hoverId === e.to;
                
                return (
                  <line
                    key={i}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={isHighlighted ? '#3B82F6' : '#6B7280'}
                    strokeWidth={isHighlighted ? 2.5 : 1.5}
                    opacity={isHighlighted ? 0.8 : 0.3}
                    markerEnd="url(#arrowhead)"
                  />
                );
              })}
            </g>
            
            <g>
              {nodes.map(n => {
                if (!n.x || !n.y) return null;
                
                const isSelected = selectedId === n.id;
                const isHovered = hoverId === n.id;
                
                let color = '#64748B';
                let icon = '●';
                if (n.type === 'user') { color = '#22C55E'; icon = '👤'; }
                if (n.type === 'ai') { color = '#3B82F6'; icon = '🤖'; }
                if (n.type === 'goal') { color = '#F59E0B'; icon = '🎯'; }
                
                const size = isHovered ? 70 : 65;
                const halfSize = size / 2;
                
                return (
                  <g 
                    key={n.id}
                    className="cursor-pointer"
                    onClick={() => setSelectedId(n.id)}
                    onMouseEnter={() => setHoverId(n.id)}
                    onMouseLeave={() => setHoverId(null)}
                    opacity={0.4 + n.confidence * 0.6}
                  >
                    <rect
                      x={n.x - halfSize}
                      y={n.y - 30}
                      width={size}
                      height={60}
                      rx={8}
                      fill="#1E293B"
                      stroke={isSelected ? color : '#374151'}
                      strokeWidth={isSelected ? 3 : 2}
                    />
                    
                    <text x={n.x} y={n.y - 35} textAnchor="middle" fontSize={18}>
                      {icon}
                    </text>
                    
                    <text
                      x={n.x}
                      y={n.y + 5}
                      textAnchor="middle"
                      fill="#E5E7EB"
                      fontSize={11}
                      fontWeight={600}
                    >
                      {n.text}
                    </text>
                    
                    <circle
                      cx={n.x + halfSize - 8}
                      cy={n.y - 20}
                      r={3 + n.confidence * 4}
                      fill={n.confidence > 0.7 ? '#22C55E' : n.confidence > 0.4 ? '#F59E0B' : '#EF4444'}
                      opacity={0.9}
                    />
                    
                    <text
                      x={n.x - halfSize + 5}
                      y={n.y + 22}
                      fontSize={9}
                      fill="#9CA3AF"
                    >
                      T{n.turn}
                    </text>
                    
                    {n.locked && (
                      <text x={n.x + halfSize - 8} y={n.y + 25} fontSize={12}>
                        🔒
                      </text>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>
      
      {/* Bottom Panel */}
      {!bottomOpen && (
        <div className="border-t border-gray-800 flex items-center justify-center h-8 hover:bg-gray-900 cursor-pointer" onClick={() => setBottomOpen(true)}>
          <ChevronUp size={14} />
        </div>
      )}
      
      {bottomOpen && (
        <div className="border-t border-gray-800 p-3 h-48 overflow-y-auto">
          {selected ? (
            <div className="flex gap-3">
              <div className="flex-1">
                <div className="flex items-center justify-between mb-2">
                  <div className="font-semibold text-sm">{selected.text}</div>
                  <button 
                    onClick={() => setBottomOpen(false)}
                    className="p-1 hover:bg-gray-800 rounded"
                  >
                    <ChevronDown size={14} />
                  </button>
                </div>
                
                <div className="flex gap-3">
                  <div className="flex-1 bg-gray-800/50 p-2 rounded">
                    <div className="text-xs text-gray-400 space-y-0.5">
                      <div>Type: <span className="text-gray-300">{selected.type}</span></div>
                      <div>Turn: <span className="text-gray-300">{selected.turn}</span></div>
                      <div>Confidence: <span className="text-gray-300">{selected.confidence.toFixed(2)}</span></div>
                      {selected.sourceSubId && (
                        <div>Source: <span className="text-blue-300">{selected.sourceSubId}</span></div>
                      )}
                    </div>
                  </div>
                  
                  <div className="flex-1 bg-gray-800/50 p-2 rounded">
                    <div className="text-xs font-semibold mb-1">Utterance</div>
                    <div className="text-xs text-gray-300">{selected.utterance || 'N/A'}</div>
                  </div>
                </div>
              </div>
              
              {selected.impulse && (
                <div className="flex-1 bg-gray-800/50 p-2 rounded">
                  <div className="text-xs font-semibold mb-1">Impulse</div>
                  <div className="flex flex-wrap gap-1">
                    {selected.impulse.semantic?.map((s, i) => (
                      <span key={i} className="px-1.5 py-0.5 bg-green-900/30 text-green-300 rounded text-xs">
                        {s}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {selected.intent && (
                <div className="flex-1 bg-gray-800/50 p-2 rounded">
                  <div className="text-xs font-semibold mb-1">Intent</div>
                  <div className="text-xs space-y-0.5">
                    <div><span className="text-gray-400">Goal:</span> {selected.intent.goal}</div>
                    <div><span className="text-gray-400">Strategy:</span> {selected.intent.strategy}</div>
                  </div>
                </div>
              )}
              
              <div className="w-32 flex flex-col gap-2">
                <button
                  onClick={() => toggleLock(selected.id)}
                  className="py-1.5 px-2 bg-gray-700 hover:bg-gray-600 rounded text-xs flex items-center justify-center gap-1"
                >
                  {selected.locked ? <Lock size={12} /> : <Unlock size={12} />}
                  {selected.locked ? 'Unlock' : 'Lock'}
                </button>
                <button
                  onClick={() => confirmNode(selected.id)}
                  className="py-1.5 px-2 bg-green-600 hover:bg-green-500 rounded text-xs"
                >
                  Confirm
                </button>
                <button
                  onClick={() => setSelectedId(null)}
                  className="py-1.5 px-2 bg-gray-700 hover:bg-gray-600 rounded text-xs"
                >
                  Close
                </button>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-between">
              <div className="text-xs text-gray-400">ノードクリックで詳細表示</div>
              <button 
                onClick={() => setBottomOpen(false)}
                className="p-1 hover:bg-gray-800 rounded"
              >
                <ChevronDown size={14} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}