import Foundation
import Testing

@testable import amanu

/// Against a real LM Studio on this Mac, and skipped when nothing answers on
/// its port: the server's behaviour (a model it does not have is answered by
/// whichever is loaded; a prompt over the context is a 400) is what the
/// backend is built around, and only the real thing can say it still holds.
struct LMStudioIntegrationTests {
    private static let baseURL = "http://127.0.0.1:1234/v1"

    /// The first loaded or listed chat model, nil when LM Studio is not there.
    private static func model() async -> String? {
        guard let url = URL(string: baseURL + "/models") else { return nil }
        var request = URLRequest(url: url)
        request.timeoutInterval = 2
        guard let (data, _) = try? await URLSession.shared.data(for: request),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = json["data"] as? [[String: Any]]
        else { return nil }
        return rows.compactMap { $0["id"] as? String }
            .first { !$0.lowercased().contains("embed") }
    }

    @Test("A real LM Studio answers through the backend, with no key")
    func answers() async throws {
        guard let model = await Self.model() else { return }
        let answer = try await LLMBackend.chatCompletion(
            backend: "lmstudio", baseURL: Self.baseURL, key: nil, model: model,
            system: "Rispondi in una riga.",
            prompt: "Riassumi: Anna e Marco rinviano il rilascio a lunedì.",
            timeout: 300, temperature: 0.2)
        #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("A prompt over the loaded context is refused with a final HTTP 400")
    func overflow() async throws {
        guard let model = await Self.model() else { return }
        let huge = String(repeating: "Riunione di prodotto, parlano di roadmap e rilasci. ", count: 6_000)
        do {
            _ = try await LLMBackend.chatCompletion(
                backend: "lmstudio", baseURL: Self.baseURL, key: nil, model: model,
                system: "Riassumi in una riga.", prompt: huge, timeout: 300)
            Issue.record("a prompt of \(huge.count) characters was accepted")
        } catch let error as LLMError {
            guard case .http(let code, _) = error else { throw error }
            #expect(code == 400)
            #expect(!error.isTransient)
        }
    }

    @Test("Configured in the file, the backend in the chain reaches LM Studio")
    func throughTheChain() async throws {
        guard let model = await Self.model() else { return }
        let home = Home(
            url: Home.sandbox().url, environment: [:], discoversTools: false, languageModels: nil)
        defer { try? FileManager.default.removeItem(at: home.url) }
        try home.writeConfig(["summary": ["backend": "lmstudio", "lmstudio_model": model]])
        let backend = try Home.$scoped.withValue(home) {
            try #require(LLMBackend.available(preference: "lmstudio").first)
        }
        #expect(backend.name == "lmstudio")
        #expect(backend.model == model)
        let answer = try await backend.call("Rispondi in una riga.", "Di' solo: pronto.")
        #expect(!answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    @Test("The listing names the loaded model with its context, and no embedding can be chosen")
    func listing() async throws {
        guard await Self.model() != nil else { return }
        let models = try await LMStudio.listModels(baseURL: Self.baseURL)
        #expect(!models.isEmpty)
        for model in models where model.type == "embeddings" {
            #expect(LMStudio.model(named: model.id, in: models) == nil)
        }
        for model in models where model.isLoaded == true {
            #expect(model.loadedContext != nil, "\(model.id) is loaded with no context")
            #expect(LMStudio.promptLimit(for: model.id, in: models)
                == LMStudio.promptLimit(contextTokens: model.loadedContext ?? 0))
        }
    }

    /// The doctor's own line, from the real `checkSummary`, against the real
    /// server — the part the fakes in `DoctorSummaryTests` cannot show.
    private static func doctorLine(model: String) throws -> Check {
        let home = Home(
            url: Home.sandbox().url, environment: [:], discoversTools: true, languageModels: nil)
        defer { try? FileManager.default.removeItem(at: home.url) }
        try home.writeConfig(["summary": ["backend": "lmstudio", "lmstudio_model": model]])
        return Home.$scoped.withValue(home) { DoctorReport.checkSummary() }
    }

    @Test("The doctor is content with a loaded model and refuses a name LM Studio would paper over")
    func doctor() async throws {
        guard await Self.model() != nil else { return }
        let models = try await LMStudio.listModels(baseURL: Self.baseURL)
        let loaded = try #require(models.first { $0.isChat && $0.isLoaded == true },
                                  "load a model in LM Studio to run this test")

        guard case .ok = try Self.doctorLine(model: loaded.id).status else {
            Issue.record("a loaded model on this Mac should be ready")
            return
        }
        let ghost = try Self.doctorLine(model: loaded.id + "-does-not-exist")
        guard case .warn(let said) = ghost.status else {
            Issue.record("a model LM Studio lacks must be reported")
            return
        }
        #expect(said.contains("has no model"))

        if let cold = models.first(where: { $0.isChat && $0.isLoaded == false }) {
            guard case .warn(let said) = try Self.doctorLine(model: cold.id).status else {
                Issue.record("a model that is not loaded should be said to load on demand")
                return
            }
            #expect(said.contains("not loaded"))
        }
    }

    /// What the Setup card says, driven through the form's own detection
    /// against the real server. `SetupWindow` starts detection in a task of
    /// its own that a test cannot wait on, so the form is asked directly.
    @Test("The Setup card reports the server's state in words")
    @MainActor
    func setupCard() async throws {
        guard await Self.model() != nil else { return }
        let models = try await LMStudio.listModels(baseURL: Self.baseURL)
        let loaded = try #require(models.first { $0.isChat && $0.isLoaded == true },
                                  "load a model in LM Studio to run this test")

        func look(_ summary: [String: Any]) async throws -> (status: String, left: String) {
            let home = Home(
                url: Home.sandbox().url, environment: [:], discoversTools: true,
                languageModels: nil)
            defer { try? FileManager.default.removeItem(at: home.url) }
            try home.writeConfig(["summary": summary])
            // The sentence is read inside the same home: outside it the form
            // would be reading the process's own config, which names nothing.
            let (form, left) = await Home.$scoped.withValue(home) {
                () async -> (SetupForm, String) in
                let form = SetupForm()
                form.reload()
                await form.detectTools()
                return (form, form.outstandingSentence)
            }
            let card = try #require(form.view.allDescendants
                .compactMap { $0 as? ChoiceCard }.first { $0.id == "lmstudio" })
            return (card.status, left)
        }
        func status(_ summary: [String: Any]) async throws -> String {
            try await look(summary).status
        }

        #expect(try await status(["backend": "lmstudio"]).hasPrefix("running · choose a model"))
        #expect(try await status(["lmstudio_model": loaded.id]) == "running · \(loaded.id)")
        #expect(try await status(["lmstudio_model": loaded.id + "-nope"]).contains("is not there"))
        #expect(try await status([
            "lmstudio_model": loaded.id, "lmstudio_base_url": "http://127.0.0.1:1/v1",
        ]) == "not answering")

        // Chosen and working, nothing is left to summarise with; chosen and
        // without a usable model, the line before Done says so.
        let missing = "something to summarise with"
        #expect(try await look([
            "backend": "lmstudio", "lmstudio_model": loaded.id,
        ]).left.contains(missing) == false)
        #expect(try await look(["backend": "lmstudio"]).left.contains(missing))
        #expect(try await look([
            "backend": "lmstudio", "lmstudio_model": loaded.id + "-nope",
        ]).left.contains(missing))
    }
}

