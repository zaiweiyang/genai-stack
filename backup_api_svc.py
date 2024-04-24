#  run "sudo /usr/bin/python3 -m pip install Flask" if install this python code as systemd service

from flask import Flask, request, jsonify
import subprocess
import datetime

app = Flask(__name__)

@app.route('/backup', methods=['POST'])
def backup():
    tag = request.json.get('tag', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        # Execute backup commands for Neo4j
        subprocess.run(f"./scripts/backup_db.sh {tag}", shell=True, check=True)
        # Execute backup for pdf_bot
        subprocess.run(f"docker exec genai-stack-pdf_bot-1 tar czf /app/data/backups/{tag}_app_data.tar.gz /app/data", shell=True, check=True)
        return jsonify({"message": "Backup successful", "tag": tag}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"message": "Backup failed", "error": str(e)}), 500

@app.route('/restore', methods=['POST'])
def restore():
    tag = request.json.get('tag')
    if not tag:
        return jsonify({"message": "Tag is required"}), 400
    try:
        # Execute restore commands for Neo4j
        subprocess.run(f"./scripts/restore_db.sh {tag}", shell=True, check=True)
        # Execute restore for pdf_bot
        subprocess.run(f"docker exec genai-stack-pdf_bot-1 tar xzf /app/data/backups/{tag}_app_data.tar.gz -C /app/data", shell=True, check=True)
        return jsonify({"message": "Restore successful"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"message": "Restore failed", "error": str(e)}), 500

@app.route('/backups', methods=['GET'])
def list_backups():
    try:
        # Execute a command within the Neo4j container to list backup directories
        result = subprocess.run(
            ["docker", "exec", "genai-stack-database-1", "bash", "-c", "ls /backups"],
            capture_output=True,
            text=True,
            check=True
        )
        backup_dirs = result.stdout.split()
        backup_dirs_sorted = sorted(backup_dirs, reverse=True)  # Latest first
        return jsonify(backup_dirs_sorted), 200
    except subprocess.CalledProcessError as e:
        # If the command failed, return the error message
        return jsonify({"message": "Failed to list backups", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
