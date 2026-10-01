graph TD
    %% Global Styling
    classDef configStyle fill:#1e1e2e,stroke:#89b4fa,stroke-width:2px,color:#cdd6f4
    classDef coreStyle fill:#181825,stroke:#f9e2af,stroke-width:2px,color:#cdd6f4
    classDef notebookStyle fill:#11111b,stroke:#a6e3a1,stroke-width:2px,color:#cdd6f4
    classDef expansionStyle fill:#181825,stroke:#f38ba8,stroke-width:2px,color:#cdd6f4

    %% ROOT PROJECT STRUCTURE
    subgraph Root ["📁 Project Root"]
        subgraph PythonCodebase ["📁 project / python"]
            EnvModule["env/
            ├── game_modes.py
            └── level_generator.py"]:::coreStyle
            
            PlannerModule["planners/
            ├── cbs.py
            ├── mstar.py
            ├── voronoi.py
            └── time_expanded.py"]:::coreStyle
            
            MarlModule["marl/
            ├── vdn.py
            ├── qmix.py
            ├── qtran.py
            └── qplex.py"]:::coreStyle
            
            ConfigModule["configs/
            └── difficulty_1.yaml ... 10.yaml"]:::configStyle
            
            UtilsModule["utils.py & ui.py
            ├── Logging & Metrics
            ├── Plotting Helpers
            └── Pygame/2D UI"]:::coreStyle
        end

        subgraph NotebookPipeline ["📁 notebooks"]
            NB1["01_environment_demo.ipynb
            • World Sanity Check
            • Heuristic / Random Agents
            • Reward & Termination Verification"]:::notebookStyle
            
            NB2["02_classical_planners.ipynb
            • CBS, M*, Voronoi Benchmarking
            • Log Path Lengths & Compute Time
            • Success Rate Evaluation"]:::notebookStyle
            
            NB3["03_marl_training.ipynb
            • Hyperparameter Management
            • Logs & Curve Inspection
            • TensorBoard / CSV Data Parsing"]:::notebookStyle
            
            NB3b["03b_curriculum_training.ipynb
            • Multi-Difficulty Sweep
            • Curriculum Progression"]:::notebookStyle
            
            NB4["04_benchmark_evaluation.ipynb
            • Joint Classical vs. MARL Comparison
            • Held-out Test Scenarios
            • Scalability & Efficiency Tracking"]:::notebookStyle
            
            NB5["05_analysis_and_plots.ipynb
            • Final Thesis Figures & Tables
            • Scaling Curves (N:M Ratios)
            • Trajectory Visualizations"]:::notebookStyle
        end

        subgraph FeatureEngine ["⚙️ Level Generation & Game Modes Engine"]
            LGParams["level_generator.py Parameters
            ├── Grid Dimensions (Rows x Cols)
            ├── Agent Ratios (Pursuers vs Evaders)
            ├── Relative Speed Ratios
            ├── Seed Control (Reproducibility)
            ├── Obstacle Density / Layout
            └── Multi-Target / Goal Coordinates"]:::expansionStyle
            
            GameVariants["game_modes.py Objectives
            ├── Pure Pursuit (Survival / Minimal Capture Time)
            ├── Reach-Avoid (Cut off path to static targets)
            └── Maze / Chokepoint (Escaping bottlenecks / Pac-Man style)"]:::expansionStyle
            
            FutureExtensions["Future Extensions
            ├── Heterogeneous / Asymmetric Roles
            ├── Fog of War / Partial Observability
            └── Interactive User-Playable UI"]:::expansionStyle
        end
    end

    %% PIPELINE WORKFLOW & DATAFLOW CONNECTIONS
    ConfigModule --> LGParams
    LGParams --> EnvModule
    GameVariants --> EnvModule

    EnvModule --> NB1
    PlannerModule --> NB2
    EnvModule --> NB2

    MarlModule --> NB3
    EnvModule --> NB3
    NB3 --> NB3b

    PlannerModule --> NB4
    MarlModule --> NB4
    EnvModule --> NB4

    NB2 --> NB5
    NB3b --> NB5
    NB4 --> NB5

    UtilsModule -. Provides Utils .-> NotebookPipeline