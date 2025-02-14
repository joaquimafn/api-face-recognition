import os
import uuid
import tempfile
import numpy as np
import face_recognition
from functools import wraps
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
load_dotenv()

ENCODINGS_DIR = 'encodings'

if not os.path.exists(ENCODINGS_DIR):
    os.makedirs(ENCODINGS_DIR)

known_face_encodings = []
known_face_names = []

def load_known_faces(encodings_dir=ENCODINGS_DIR):
    """
    Carrega todas as codificações salvas em arquivos .npy do diretório especificado.
    
    Retorna duas listas:
      - known_face_encodings: lista com os arrays NumPy de codificações.
      - known_face_names: lista com os nomes (extraídos do nome do arquivo, sem a extensão).
    """
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(encodings_dir):
        print(f"Diretório '{encodings_dir}' não encontrado.")
        return known_face_encodings, known_face_names

    for file_name in os.listdir(encodings_dir):
        if file_name.endswith('.npy'):
            file_path = os.path.join(encodings_dir, file_name)
            encoding = np.load(file_path)
            known_face_encodings.append(encoding)
            # O nome é o nome do arquivo sem a extensão
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
    return known_face_encodings, known_face_names

load_known_faces()

AUTHORIZED_API_KEY = os.getenv('AUTHORIZED_API_KEY')

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != AUTHORIZED_API_KEY:
            return jsonify({'error': 'Unauthorized'}), 403
        return f(*args, **kwargs)
    return decorated_function


# @app.after_request
# def set_security_headers(response):
#     response.headers['Content-Security-Policy'] = "default-src 'self'"
#     response.headers['X-Content-Type-Options'] = 'nosniff'
#     response.headers['X-Frame-Options'] = 'DENY'
#     response.headers['Referrer-Policy'] = 'no-referrer'
#     # Opcionalmente, force HTTPS (HSTS) se já estiver operando com TLS:
#     response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
#     return response

@app.route("/recognize", methods=["POST"])
@require_api_key
def recognize():
    known_face_encodings, known_face_names = load_known_faces()

    for file_name in os.listdir(ENCODINGS_DIR):
        if file_name.endswith('.npy'):
            file_path = os.path.join(ENCODINGS_DIR, file_name)
            encoding = np.load(file_path)
            known_face_encodings.append(encoding)

            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)

    if not known_face_encodings:
        return jsonify({
            'error': 'Nenhuma codificação facial cadastrada. Cadastre um rosto primeiro'
        }), 400

    if 'image' not in request.files:
        return jsonify({"error": "Imagem não fornecida. Use o campo 'image'."}), 400

    file = request.files['image']

    temp_file_path = os.path.join(tempfile.gettempdir(), file.filename)
    file.save(temp_file_path)

    image = face_recognition.load_image_file(temp_file_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            results.append({
                "name": name,
                "recognized": True,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left}
            })
        else:
            results.append({
                "recognized": False,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left}
            })

    os.remove(temp_file_path)

    return jsonify(results)

@app.route('/save_encoding', methods=['POST'])
@require_api_key
def save_encoding():
    if 'image' not in request.files:
        return jsonify({'error': 'Arquivo de imagem não fornecido. Use o campo "image".'}), 400

    image_file = request.files['image']

    temp_file_path = os.path.join(tempfile.gettempdir(), image_file.filename)
    image_file.save(temp_file_path)

    image = face_recognition.load_image_file(temp_file_path)
    face_encodings = face_recognition.face_encodings(image)

    os.remove(temp_file_path)

    if not face_encodings:
        return jsonify({'error': 'Nenhum rosto detectado na imagem.'}), 400
    
    new_encoding = face_recognition.face_encodings(image)[0]

    for file_name in os.listdir(ENCODINGS_DIR):
        if file_name.endswith('.npy'):
            file_path = os.path.join(ENCODINGS_DIR, file_name)
            existing_encoding = np.load(file_path)

            duplicate = face_recognition.compare_faces([existing_encoding], new_encoding, tolerance=0.6)
            if duplicate[0]:
                return jsonify({
                    'error': 'Imagem já está cadastrada.',
                    'filename': file_name
                }), 400

    encoding = new_encoding

    name = request.form.get('name')
    document = request.form.get('document')
    if not name:
        name = str(uuid.uuid4())

    if not document:
        npy_filename = f"{name}.npy"
    else:
        npy_filename = f"{name}-{document}.npy"

    npy_path = os.path.join(ENCODINGS_DIR, npy_filename)
    np.save(npy_path, encoding)

    return jsonify({
        'message': 'Codificação facial salva com sucesso!',
        'filename': npy_filename
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
