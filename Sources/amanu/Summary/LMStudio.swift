import Foundation

/// What Amanu knows about LM Studio that is not in the OpenAI shape it shares
/// with other servers. The chat call itself is `LLMBackend.chatCompletion`.
enum LMStudio {
    /// A model LM Studio has, as far as its listing says.
    struct Model: Equatable, Sendable {
        let id: String
        /// `llm`, `vlm` or `embeddings`; nil from the plain OpenAI listing,
        /// which does not say.
        let type: String?
        /// Whether the model is in memory now; nil when the listing does not
        /// say. A model that is not is loaded on demand, if the person has
        /// allowed that, with a smaller context than they may have chosen.
        let isLoaded: Bool?
        /// The context the loaded model was given, in tokens.
        let loadedContext: Int?
        let maxContext: Int?

        /// Whether it can be asked for a summary. An embedding model cannot,
        /// and LM Studio will not say so: it answers with another model.
        var isChat: Bool { type != "embeddings" }
    }

    /// The prompt, in characters, a backend is trusted to read whole before
    /// the real context of the loaded model is known.
    ///
    /// LM Studio loads a model with the context chosen in its own window, and
    /// a model it loads on demand gets 8,192 tokens — measured against LM
    /// Studio on a Mac whose hand-loaded model had 26,923. A request over the
    /// context is refused with HTTP 400 rather than cut, which is better than
    /// Ollama's quiet truncation but still a summary lost, so callers split a
    /// transcript to this size first. Room is left for the reasoning a model
    /// may do before it answers, which counts against the same context: a
    /// 12B model spent 208 of 226 tokens on it for a one-line summary. Meetings
    /// here are mostly Russian, about two characters a token.
    static let defaultPromptLimit = promptLimit(contextTokens: 8_192)

    /// Room for the answer, and for reasoning before it.
    static let replyTokens = 4_096

    /// The least a context may be trusted to hold, so a tiny one still makes
    /// progress in chunks rather than a limit of nothing.
    private static let floorCharacters = 2_000

    static func promptLimit(contextTokens: Int) -> Int {
        max(floorCharacters, (contextTokens - replyTokens) * 2)
    }

    /// The prompt limit for `model` given what the server lists: from the
    /// context it is loaded with when it is loaded, the default otherwise.
    static func promptLimit(for model: String?, in models: [Model]?) -> Int {
        guard let model, let found = models?.first(where: { $0.id == model }),
              found.isLoaded == true, let context = found.loadedContext
        else { return defaultPromptLimit }
        return promptLimit(contextTokens: context)
    }

    /// The model the person named, if the server has it and it can chat.
    /// "Has it" is checked rather than assumed because LM Studio answers a
    /// request for a model it does not have with whichever one is loaded.
    static func model(named name: String, in models: [Model]) -> Model? {
        models.first { $0.id == name && $0.isChat }
    }

    // MARK: - listing

    /// The server's root, which is the Base URL without its `/v1`: LM Studio's
    /// own listing, the one that says what is loaded, lives beside it.
    static func serverRoot(baseURL: String) -> String {
        var base = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        if base.lowercased().hasSuffix("/v1") { base.removeLast(3) }
        return base
    }

    /// What LM Studio has, with what is loaded and how big its context is.
    ///
    /// Asks LM Studio's own listing first and falls back to the OpenAI one,
    /// which names every downloaded model — embeddings included — and says
    /// nothing of what is loaded. The own listing is a beta API: a version
    /// without it, or one that changes its shape, still gets a working list.
    static func listModels(
        baseURL: String, key: String? = nil, timeout: TimeInterval = 2,
        session: URLSession = .shared
    ) async throws -> [Model] {
        if let own = try? await fetch(
            try OpenAICompatible.url(baseURL: serverRoot(baseURL: baseURL), path: "api/v0/models"),
            key: key, timeout: timeout, session: session),
           let models = try? models(fromOwnListing: own) {
            return models
        }
        let plain = try await fetch(
            try OpenAICompatible.url(baseURL: baseURL, path: "models"),
            key: key, timeout: timeout, session: session)
        return try models(fromOpenAIListing: plain)
    }

    private static func fetch(
        _ url: URL, key: String?, timeout: TimeInterval, session: URLSession
    ) async throws -> Data {
        var request = URLRequest(url: url)
        request.timeoutInterval = timeout
        if let key { request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization") }
        let (data, response) = try await session.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            throw LLMError.http(status, String(decoding: data.prefix(400), as: UTF8.self))
        }
        return data
    }

    static func models(fromOwnListing data: Data) throws -> [Model] {
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rows = json["data"] as? [[String: Any]]
        else { throw LLMError.malformedResponse("lmstudio") }
        return rows.compactMap { row in
            guard let id = row["id"] as? String else { return nil }
            return Model(
                id: id, type: row["type"] as? String,
                isLoaded: (row["state"] as? String).map { $0 == "loaded" },
                loadedContext: (row["loaded_context_length"] as? NSNumber)?.intValue,
                maxContext: (row["max_context_length"] as? NSNumber)?.intValue)
        }
    }

    static func models(fromOpenAIListing data: Data) throws -> [Model] {
        guard
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let rows = json["data"] as? [[String: Any]]
        else { throw LLMError.malformedResponse("lmstudio") }
        return rows.compactMap { row in
            guard let id = row["id"] as? String else { return nil }
            // The plain listing does not say what a model is, and an
            // embedding model cannot write a summary; its name is all there is
            // to tell it by.
            let looksLikeEmbedding = id.lowercased().contains("embed")
            return Model(
                id: id, type: looksLikeEmbedding ? "embeddings" : nil,
                isLoaded: nil, loadedContext: nil, maxContext: nil)
        }
    }
}
