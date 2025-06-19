from flask import Flask, render_template, request
from qna_summarizer import summarize_questions

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        questions = request.form["questions"]
        n_clusters_str = request.form.get("n_clusters", "").strip()
        n_clusters = int(n_clusters_str) if n_clusters_str.isdigit() else None

        question_list = [q.strip() for q in questions.split("\n") if q.strip()]
        summary = summarize_questions(question_list, n_clusters=n_clusters)
        return render_template("result.html", summary=summary)

    return render_template("index.html")

if __name__ == "__main__":
    print("Flask アプリを起動中... http://127.0.0.1:5001 でアクセスできます")
    app.run(host="0.0.0.0", port=5001, debug=True)
