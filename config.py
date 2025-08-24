APP_VERSION = "v1.0"

# LLM Model Names
GEMINI_MODEL = "gemini-1.5-flash"
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-haiku-20240307"
OLLAMA_MODEL = "llama3.1"
DATABASE_FILE = "resume_suite.db"

# Resume Writing Clichés and Buzzwords to detect
RESUME_CLICHES = {
    "team player", "results-oriented", "go-getter", "synergy", "hard worker",
    "self-motivated", "detail-oriented", "proactive", "dynamic", "out-of-the-box thinker",
}

# Action verbs for resume analysis
ACTION_VERBS = {
    'achieved', 'accelerated', 'administered', 'advised', 'advocated', 'analyzed', 'authored',
    'automated', 'built', 'calculated', 'centralized', 'chaired', 'coached', 'collaborated',
    'conceived', 'consolidated', 'constructed', 'consulted', 'converted', 'coordinated',
    'created', 'debugged', 'decreased', 'defined', 'delivered', 'designed', 'developed',
    'directed', 'documented', 'drove', 'eliminated', 'engineered', 'enhanced', 'established',
    'evaluated', 'executed', 'expanded', 'facilitated', 'founded', 'generated', 'grew',
    'guided', 'identified', 'implemented', 'improved', 'increased', 'influenced', 'initiated',
    'innovated', 'inspired', 'integrated', 'interpreted', 'invented', 'launched', 'led',
    'managed', 'mastered', 'mentored', 'modernized', 'motivated', 'negotiated', 'optimized',
    'orchestrated', 'overhauled', 'owned', 'pioneered', 'planned', 'prioritized', 'produced',
    'proposed', 'quantified', 'ran', 'rebuilt', 'reduced', 're-engineered', 'resolved',
    'restructured', 'revamped', 'saved', 'scaled', 'shipped', 'simplified', 'solved',
    'spearheaded', 'standardized', 'streamlined', 'strengthened', 'succeeded', 'supervised',
    'taught', 'trained', 'transformed', 'unified', 'won', 'wrote'
}