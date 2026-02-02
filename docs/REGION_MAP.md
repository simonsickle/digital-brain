# Region Map

This map summarizes how brain regions cooperate during a full cognitive cycle.

## High-Level Connectivity

```mermaid
graph LR
    Thalamus((Thalamus))
    Amygdala((Amygdala))
    Prefrontal((Prefrontal))
    Hippocampus((Hippocampus))
    DMN((DMN))
    Workspace((Workspace))
    Language((Language Cortex))
    Temporal((Temporal Cortex))
    PPC((Posterior Parietal))
    Motor((Motor Cortex))
    Basal((Basal Ganglia))
    Cerebellum((Cerebellum))
    Insula((Insula))
    Hypo((Hypothalamus))

    Thalamus --> Amygdala
    Thalamus --> PPC
    Thalamus --> Language
    PPC --> Workspace
    Amygdala --> Workspace
    Language --> Temporal
    Language --> Prefrontal
    Temporal --> Prefrontal
    Temporal --> DMN
    Workspace --> Prefrontal
    Workspace --> Hippocampus
    Prefrontal --> Basal
    Basal --> Motor
    Motor --> Cerebellum
    Insula --> Workspace
    Hypo --> Insula
    Hippocampus --> DMN
```

## Signal Types by Region

| Region | Primary Signal Types | Role |
| --- | --- | --- |
| Thalamus | Sensory, Attention | Gating + routing |
| Amygdala | Emotion, Attention | Valence + arousal tagging |
| Posterior Parietal | Sensory | Multimodal binding |
| Language Cortex | Sensory, Memory | Linguistic grounding |
| Temporal Cortex | Memory | Semantic association |
| Prefrontal | Memory, Attention | Working memory + executive control |
| Workspace | Broadcast | Conscious access |
| Hippocampus | Memory | Encoding + consolidation |
| DMN | Memory, Broadcast | Self-model + narrative |
| Basal Ganglia | Motor | Action selection |
| Motor Cortex | Motor | Motor planning + execution |
| Cerebellum | Motor | Timing + procedural refinement |
| Insula | Sensory, Attention | Interoception + body state |
| Hypothalamus | Emotion | Homeostasis + drive regulation |

## Cycle Snapshot (Mermaid)

```mermaid
sequenceDiagram
    participant Input
    participant Thalamus
    participant Amygdala
    participant Language
    participant Temporal
    participant Workspace
    participant Prefrontal
    participant Hippocampus

    Input->>Thalamus: Sensory signal
    Thalamus->>Amygdala: Emotional tagging
    Thalamus->>Language: Textual parsing
    Language->>Temporal: Semantic association
    Amygdala->>Workspace: Salience boost
    Temporal->>Workspace: Semantic insight
    Workspace->>Prefrontal: Working memory update
    Workspace->>Hippocampus: Memory encoding
```
