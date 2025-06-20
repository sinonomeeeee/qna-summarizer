def summarize_questions(questions, n_clusters=13):
    if not questions:
        return []

    n_clusters = min(n_clusters, len(questions))

    embeddings = get_sentence_embeddings(questions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clustered = {}
    clustered_embeddings = {}

    for label, question, embedding in zip(labels, questions, embeddings):
        clustered.setdefault(label, []).append(question)
        clustered_embeddings.setdefault(label, []).append(embedding)

    summary = []
    for cluster_id in clustered:
        cluster_questions = clustered[cluster_id]
        cluster_embeds = np.array(clustered_embeddings[cluster_id])
        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(cluster_embeds - centroid, axis=1)
        representative_idx = np.argmin(distances)

        summary.append({
            "cluster": cluster_id + 1,
            "representative": cluster_questions[representative_idx],
            "questions": cluster_questions
        })

    return summary



