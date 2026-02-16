# Resume Analyzer & Job Match System

AI-powered resume analysis tool using NLP and deep learning to compare resumes with job descriptions and provide detailed matching scores.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Deployment](#deployment)
- [Author](#author)

## 🎯 Overview

This application analyzes resumes against job descriptions using state-of-the-art NLP models to provide:
- Overall compatibility scores
- Section-by-section analysis
- Keyword matching
- Skill gap identification
- Improvement suggestions

Built with Gradio for an interactive web interface and optimized for deployment on Hugging Face Spaces.

## ✨ Features

- **Multi-Model Analysis**: Uses BERT, Sentence Transformers, and TF-IDF for comprehensive matching
- **Document Support**: Accepts PDF and DOCX formats for both resumes and job descriptions
- **Detailed Scoring**: Provides scores for:
  - Overall match percentage
  - Skills alignment
  - Experience relevance
  - Education compatibility
  - Keyword density
- **Visual Feedback**: Generates word clouds and similarity visualizations
- **API Support**: FastAPI endpoints for programmatic access
- **Cloud-Ready**: Optimized for Hugging Face Spaces deployment

## 🛠 Technology Stack

### Core ML/NLP
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - BERT models for contextual understanding
- **Sentence Transformers** - Semantic similarity with \`all-MiniLM-L6-v2\`
- **Scikit-learn** - TF-IDF vectorization and cosine similarity

### Document Processing
- **PyMuPDF (fitz)** - PDF text extraction
- **python-docx** - Word document processing

### Web Framework
- **Gradio** - Interactive web UI
- **FastAPI** - REST API endpoints
- **Uvicorn** - ASGI server

### Visualization
- **Matplotlib** - Plotting and charts
- **WordCloud** - Visual keyword representation

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for transformer models)

### Setup

1. Clone the repository:
\`\`\`bash
git clone https://github.com/pradyten/Resume-Comparator.git
cd Resume-Comparator
\`\`\`

2. Create a virtual environment (recommended):
\`\`\`bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

3. Install dependencies:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

**Note:** The installation may take several minutes as it downloads pre-trained transformer models (~400MB).

## 👨‍💻 Author

**Pradyumn Tendulkar**

Data Science Graduate Student | ML Engineer

- GitHub: [@pradyten](https://github.com/pradyten)
- LinkedIn: [Pradyumn Tendulkar](https://www.linkedin.com/in/p-tendulkar/)
- Email: pktendulkar@wpi.edu

---

⭐ If you found this project helpful, please consider giving it a star!

📝 **License:** MIT

💡 **Contributing:** Pull requests are welcome! For major changes, please open an issue first to discuss proposed changes.
