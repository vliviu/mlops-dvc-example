#!/usr/bin/env bash
# ===========================================================
# setup_dvc_project.sh
# Creates an MLOps+DVC project skeleton with local storage
# ===========================================================

# Exit immediately on errors
set -e

# === CONFIGURATION ===
PROJECT_NAME="mlops-dvc-example"
LOCAL_DVC_REMOTE="$HOME/BankGroz/dvc-storage"   # <-- change if needed

# === CREATE FOLDER STRUCTURE ===
echo "Creating project structure..."
mkdir -p ${PROJECT_NAME}/{data,src,models}
mkdir -p ${LOCAL_DVC_REMOTE}

# === CREATE SOURCE FILES ===
echo "📄 Creating source files..."
touch ${PROJECT_NAME}/src/{prepare.py,featurize.py,train.py}

# === CREATE CONFIG / METADATA FILES ===
touch ${PROJECT_NAME}/{params.yaml,dvc.yaml,dvc.lock,metrics.yaml,README.md,requirements.txt}

# === INITIALIZE GIT & DVC ===
cd ${PROJECT_NAME}
echo "⚙️ Initializing Git and DVC..."
git init -q
dvc init -q

# === ADD LOCAL DVC REMOTE ===
echo "🔗 Setting local DVC remote storage..."
dvc remote add -d local_store "${LOCAL_DVC_REMOTE}"
git add .dvc .gitignore
git commit -m "Initialize DVC project with local remote storage"

# === CONFIRM STRUCTURE ===
echo "✅ Project setup complete!"
echo
tree -L 2
echo
echo "DVC remote location: ${LOCAL_DVC_REMOTE}"
echo "Next steps:"
echo "1️⃣ cd ${PROJECT_NAME}"
echo "2️⃣ Add your data and scripts"
echo "3️⃣ Run: dvc repro"
