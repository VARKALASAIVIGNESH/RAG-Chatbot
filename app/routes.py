from flask import Blueprint, render_template, request, jsonify
from .rag_engine import get_rag_response

main = Blueprint("main", __name__)

@main.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form.get("query")
        if not user_input:
            return jsonify({"error": "No query provided"}), 400
        response = get_rag_response(user_input)
        return jsonify({"response": response})
    return render_template("index.html")
