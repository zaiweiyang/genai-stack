#  run "sudo /usr/bin/python3 -m pip install Flask" if install this python code as systemd service

from flask import Flask, request, jsonify
import subprocess
import datetime

import logging
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app_data_backup_path = "/app/data_backups"

@app.route('/backup', methods=['POST'])
def backup():
    global app_data_backup_path
    tag = request.json.get('tag', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    try:
        logging.info(f"Starting backup with tag: {tag}")
        # Check if the backup file already exists
        backup_path = f"{app_data_backup_path}/{tag}_app_data.tar.gz"
        check_command = f"docker exec genai-stack-pdf_bot-1 ls {backup_path}"
        check_result = subprocess.run(check_command, shell=True, capture_output=True, text=True)

        if check_result.returncode == 0:
            logging.warning(f"Backup with tag {tag} already exists. Skipping backup.")
            return jsonify({"message": "Backup already exists", "tag": tag}), 200

        # Execute backup commands for Neo4j
        subprocess.run(f"./scripts/backup_db.sh {tag}", shell=True, check=True)
        # Execute backup for pdf_bot
        # subprocess.run(f"docker exec genai-stack-pdf_bot-1 tar czf /app/data/backups/{tag}_app_data.tar.gz /app/data", shell=True, check=True)
        result = subprocess.run(f"docker exec genai-stack-pdf_bot-1 tar czf {backup_path} -C / app/data", shell=True, check=True)
        if result.returncode != 0:
            logging.error(f"Backup failed: {result.stderr}")
            return jsonify({"message": "Backup failed", "error": result.stderr}), 500
        
        logging.info(f"Backup command output: {result.stdout}")

        # Verify the tar.gz file was created
        check_command = f"docker exec genai-stack-pdf_bot-1 ls {backup_path}"
        check_result = subprocess.run(check_command, shell=True, capture_output=True, text=True)
        if check_result.returncode == 0:
            logging.info(f"Backup successful with tag: {tag}")
            return jsonify({"message": "Backup successful", "tag": tag}), 200
        else:
            logging.error("Backup file is missing or empty.")
            return jsonify({"message": "Backup file is missing or empty."}), 500

    except subprocess.CalledProcessError as e:
        return jsonify({"message": "Backup failed", "error": str(e)}), 500

@app.route('/restore', methods=['POST'])
def restore():
    global app_data_backup_path
    tag = request.json.get('tag')
    if not tag:
        return jsonify({"message": "Tag is required"}), 400
    try:
        logging.info(f"Starting restore with tag: {tag}")
        backup_path = f"{app_data_backup_path}/{tag}_app_data.tar.gz"
        check_command = f"docker exec genai-stack-pdf_bot-1 ls {backup_path}"
        check_result = subprocess.run(check_command, shell=True, capture_output=True, text=True)
        
        if check_result.returncode != 0:
            logging.error(f"Restore failed: Backup file {backup_path} does not exist")
            return jsonify({"message": f"Backup file {backup_path} does not exist"}), 400

        # Execute restore commands for Neo4j
        subprocess.run(f"./scripts/restore_db.sh {tag}", shell=True, check=True)

          # Verify the tar.gz file is not corrupted
        logging.info(f"Verifying the integrity of the tarball {backup_path}")
        verify_command = f"docker exec genai-stack-pdf_bot-1 tar tzf {backup_path}"
        verify_result = subprocess.run(verify_command, shell=True, capture_output=True, text=True)
        
        if verify_result.returncode != 0:
            logging.error(f"Restore failed: Backup file {backup_path} is corrupted")
            return jsonify({"message": f"Backup file {backup_path} is corrupted", "error": verify_result.stderr}), 500

         # Clear the target directory before restoring
        logging.info("Clearing the target directory before restoring")
        clear_command = f"docker exec genai-stack-pdf_bot-1 rm -f /app/data/upload_status.json"
        clear_result = subprocess.run(clear_command, shell=True, capture_output=True, text=True)
        if clear_result.returncode != 0:
            logging.error(f"Failed to clear target directory: {clear_result.stderr}")
            return jsonify({"message": "Failed to clear target directory", "error": clear_result.stderr}), 500
        # Execute restore for pdf_bot
        # subprocess.run(f"docker exec genai-stack-pdf_bot-1 tar xzf /app/data/backups/{tag}_app_data.tar.gz -C /app/data", shell=True, check=True)
        restore_command = f"docker exec genai-stack-pdf_bot-1 tar xzf {backup_path} -C /"
        result = subprocess.run(restore_command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"Restore failed: {result.stderr}")
            return jsonify({"message": "Restore failed", "error": result.stderr}), 500
        
        logging.info(f"Restore command output: {result.stdout}")
        logging.info(f"Restore successful with tag: {tag}")
        return jsonify({"message": f"Restore successful with tag: {tag}"}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({"message": "Restore failed", "error": str(e)}), 500

@app.route('/backups', methods=['GET'])
def list_backups():
    try:
        # Execute a command within the Neo4j container to list backup directories with timestamps
        command = ["docker", "exec", "genai-stack-database-1", "bash", "-c", "ls -lt --time-style=long-iso /backups"]
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        backup_dirs = []
        for line in lines[1:]:  # Skip the first line as it contains the total
            parts = line.split()
            if len(parts) > 5:
                backup_dirs.append(parts[-1])  # The last part is the directory name

        return jsonify(backup_dirs), 200
    except subprocess.CalledProcessError as e:
        # If the command failed, return the error message
        return jsonify({"message": "Failed to list backups", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
