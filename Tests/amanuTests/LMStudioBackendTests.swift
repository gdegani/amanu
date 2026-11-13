import Foundation
import Testing

@testable import amanu

struct LMStudioBackendTests {
    @Test("LM Studio is asked without an Authorization header when it wants no token")
    func requestWithoutKey() throws {
        let request = try LLMBackend.chatCompletionRequest(
            baseURL: "http://127.0.0.1:1234/v1", key: nil, model: "google/gemma-4-12b",
            system: "Take meeting notes.", prompt: "me: ship it", timeout: 1_800,
            temperature: 0.2)

        #expect(request.url?.absoluteString == "http://127.0.0.1:1234/v1/chat/completions")
        #expect(request.value(forHTTPHeaderField: "authorization") == nil)
        let data = try #require(request.httpBody)
        let body = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(body["model"] as? String == "google/gemma-4-12b")
        #expect(body["temperature"] as? Double == 0.2)
    }

    @Test("A token, when there is one, goes as a bearer; OpenAI's own request sends no temperature")
    func requestWithKey() throws {
        let request = try LLMBackend.chatCompletionRequest(
            baseURL: "http://127.0.0.1:1234/v1", key: "t0ken", model: "m",
            system: "s", prompt: "p", timeout: 600)
        #expect(request.value(forHTTPHeaderField: "authorization") == "Bearer t0ken")
        let data = try #require(request.httpBody)
        let body = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(body["temperature"] == nil)
    }

    @Test("Plain http to another machine is refused before anything is sent")
    func remoteHTTPRefused() {
        #expect(throws: OpenAICompatible.EndpointError.self) {
            _ = try LLMBackend.chatCompletionRequest(
                baseURL: "http://studio.local:1234/v1", key: nil, model: "m",
                system: "s", prompt: "p", timeout: 1)
        }
    }

    @Test("The answer is the content; the reasoning beside it is not read")
    func responseParsing() throws {
        let ok = Data(###"{"choices":[{"message":{"role":"assistant","content":"## Decisions","reasoning_content":"private"}}]}"###.utf8)
        #expect(try LLMBackend.chatCompletionText(
            from: ok, statusCode: 200, backend: "lmstudio", requireText: true) == "## Decisions")

        let empty = Data(#"{"choices":[{"message":{"content":"  "}}]}"#.utf8)
        #expect(throws: LLMError.self) {
            _ = try LLMBackend.chatCompletionText(
                from: empty, statusCode: 200, backend: "lmstudio", requireText: true)
        }
    }

    @Test("A context too small is an HTTP 400 that is not retried")
    func contextOverflowIsFinal() throws {
        let body = Data(##"{"error":"The number of tokens to keep from the initial prompt is greater than the context length."}"##.utf8)
        let error = try #require(throws: LLMError.self) {
            _ = try LLMBackend.chatCompletionText(
                from: body, statusCode: 400, backend: "lmstudio", requireText: true)
        }
        #expect(!error.isTransient)
    }

    @Test("A configured LM Studio with no model chosen says which setting is missing")
    func noModel() {
        #expect(!LLMError.notConfigured("lmstudio", "summary.lmstudio_model").isTransient)
        #expect("\(LLMError.notConfigured("lmstudio", "summary.lmstudio_model"))"
            .contains("summary.lmstudio_model"))
    }

    @Test("Settings: LM Studio is unconfigured until the file says something about it")
    func settings() {
        let none = Config.summary(in: nil)
        #expect(!none.lmstudioConfigured)
        #expect(none.lmstudioModel == nil)
        #expect(none.lmstudioBaseURL == "http://127.0.0.1:1234/v1")

        let named = Config.summary(in: ["summary": ["lmstudio_model": " google/gemma-4-12b "]])
        #expect(named.lmstudioConfigured)
        #expect(named.lmstudioModel == "google/gemma-4-12b")
    }

    @Test("LM Studio joins auto only when configured, and ahead of the Ollama nobody chose")
    func chain() throws {
        let home = Home(
            url: Home.sandbox().url, environment: [:], discoversTools: false, languageModels: nil)
        defer { try? FileManager.default.removeItem(at: home.url) }
        try Home.$scoped.withValue(home) {
            #expect(LLMBackend.available(preference: "auto").map(\.name) == ["ollama"])

            try home.writeConfig(["summary": ["lmstudio_model": "google/gemma-4-12b"]])
            let auto = LLMBackend.available(preference: "auto")
            #expect(auto.map(\.name) == ["lmstudio", "ollama"])
            #expect(auto.first?.isUnchosenFallback == false)
            #expect(auto.first?.model == "google/gemma-4-12b")
            #expect(LLMBackend.available(preference: "lmstudio").map(\.name) == ["lmstudio"])
        }
    }

    @Test("The token is read from its own file and never from the OpenAI key")
    func tokenIsItsOwn() throws {
        let home = Home(
            url: Home.sandbox().url, environment: ["OPENAI_API_KEY": "sk-openai"],
            discoversTools: false, languageModels: nil)
        defer { try? FileManager.default.removeItem(at: home.url) }
        try Home.$scoped.withValue(home) {
            #expect(Credentials.lmstudioKey() == nil)

            let file = home.url.appendingPathComponent("lm-token")
            try "lm-secret\n".write(to: file, atomically: true, encoding: .utf8)
            try home.writeConfig(["summary": ["lmstudio_api_key_path": file.path]])
            #expect(Credentials.lmstudioKey() == "lm-secret")
        }
    }
}

struct LMStudioEgressTests {
    @Test("A job that failed for good is offered again when LM Studio's address or token changes")
    func fingerprintFollowsLMStudio() throws {
        let home = Home(
            url: Home.sandbox().url, environment: [:], discoversTools: false, languageModels: nil)
        defer { try? FileManager.default.removeItem(at: home.url) }
        try Home.$scoped.withValue(home) {
            try home.writeConfig(["summary": ["backend": "lmstudio", "lmstudio_model": "m"]])
            let before = try #require(MeetingEgress.fingerprint(for: .summary))

            try home.writeConfig(["summary": [
                "backend": "lmstudio", "lmstudio_model": "m",
                "lmstudio_base_url": "http://127.0.0.1:4321/v1",
            ]])
            let moved = try #require(MeetingEgress.fingerprint(for: .summary))
            #expect(moved != before)

            let file = home.url.appendingPathComponent("lm-token")
            try "secret\n".write(to: file, atomically: true, encoding: .utf8)
            try home.writeConfig(["summary": [
                "backend": "lmstudio", "lmstudio_model": "m",
                "lmstudio_base_url": "http://127.0.0.1:4321/v1",
                "lmstudio_api_key_path": file.path,
            ]])
            #expect(MeetingEgress.fingerprint(for: .summary) != moved)
        }
    }
}
