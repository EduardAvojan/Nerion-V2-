# Installation

This project uses a modern, flexible dependency management system. You can choose to install only the components you need.

### 1. Clone the Repository

```bash
git clone https://github.com/EduardAvojan/Nerion-V2-.git
cd Nerion-V2-
```

### 2. Install Dependencies

Choose one of the following installation methods:

**A) Minimal Installation (Core Agent)**

This is recommended for headless environments or for users who only need the core AI and automation logic.

```bash
pip install -e .
```

**B) Full User Installation (with Voice and Web)**

This installs the core agent plus all dependencies required for voice interaction and web browsing capabilities.

```bash
pip install -e ".[voice,web,docs]"
```

**C) Development Installation**

This installs all dependencies, including the tools required for testing and development.

```bash
pip install -e ".[dev,voice,web,docs]"
```

### 3. Provide API Credentials

Create a `.env` file for your API keys by copying the example template.

```bash
cp .env.example .env
```

Now, edit the `.env` file and add your API keys (e.g., `NERION_V2_GEMINI_KEY`).
