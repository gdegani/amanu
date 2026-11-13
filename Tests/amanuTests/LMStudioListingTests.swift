import Foundation
import Testing

@testable import amanu

struct LMStudioListingTests {
    private static let own = Data(###"""
    {"data":[
      {"id":"google/gemma-4-12b","type":"vlm","state":"loaded","max_context_length":262144,"loaded_context_length":26923},
      {"id":"qwen2.5-coder-14b-instruct-mlx","type":"llm","state":"not-loaded","max_context_length":32768},
      {"id":"nomic-ai/text-embedding-nomic-embed-text-v1.5","type":"embeddings","state":"not-loaded"}
    ],"object":"list"}
    """###.utf8)

    @Test("LM Studio's own listing says what is loaded and with how much context")
    func ownListing() throws {
        let models = try LMStudio.models(fromOwnListing: Self.own)
        #expect(models.map(\.id) == [
            "google/gemma-4-12b", "qwen2.5-coder-14b-instruct-mlx",
            "nomic-ai/text-embedding-nomic-embed-text-v1.5",
        ])
        #expect(models[0].isLoaded == true)
        #expect(models[0].loadedContext == 26_923)
        #expect(models[1].isLoaded == false)
        #expect(models[1].loadedContext == nil)
    }

    @Test("The plain OpenAI listing says nothing of state, and embeddings are told by name")
    func openAIListing() throws {
        let data = Data(###"{"data":[{"id":"google/gemma-4-12b"},{"id":"nomic-ai/text-embedding-nomic-embed-text-v1.5"}]}"###.utf8)
        let models = try LMStudio.models(fromOpenAIListing: data)
        #expect(models[0].isLoaded == nil)
        #expect(models[0].isChat)
        #expect(!models[1].isChat)
    }

    @Test("A model is the one named only if the server has it and it can chat")
    func namedModel() throws {
        let models = try LMStudio.models(fromOwnListing: Self.own)
        #expect(LMStudio.model(named: "google/gemma-4-12b", in: models) != nil)
        #expect(LMStudio.model(named: "google/gemma-4-12", in: models) == nil)
        #expect(LMStudio.model(
            named: "nomic-ai/text-embedding-nomic-embed-text-v1.5", in: models) == nil)
    }

    @Test("The prompt limit follows the context the model was loaded with")
    func promptLimit() throws {
        let models = try LMStudio.models(fromOwnListing: Self.own)
        #expect(LMStudio.promptLimit(for: "google/gemma-4-12b", in: models)
            == (26_923 - LMStudio.replyTokens) * 2)
        // Not loaded: on-demand loading gave 8,192 tokens when measured.
        #expect(LMStudio.promptLimit(for: "qwen2.5-coder-14b-instruct-mlx", in: models)
            == LMStudio.defaultPromptLimit)
        #expect(LMStudio.promptLimit(for: nil, in: nil) == LMStudio.defaultPromptLimit)
        #expect(LMStudio.promptLimit(contextTokens: 1_000) == 2_000)
    }

    @Test("The own listing is beside /v1, not under it")
    func serverRoot() {
        #expect(LMStudio.serverRoot(baseURL: "http://127.0.0.1:1234/v1") == "http://127.0.0.1:1234")
        #expect(LMStudio.serverRoot(baseURL: " http://127.0.0.1:1234/v1/ ") == "http://127.0.0.1:1234")
        #expect(LMStudio.serverRoot(baseURL: "https://proxy.example/lm/v1") == "https://proxy.example/lm")
        #expect(LMStudio.serverRoot(baseURL: "http://127.0.0.1:1234") == "http://127.0.0.1:1234")
    }

    @Test("A backend asks the server for its limit, and falls back to the default")
    func effectiveLimit() async {
        var backend = LLMBackend(name: "x", model: nil, promptLimit: 100) { _, _ in "" }
        #expect(await backend.effectivePromptLimit() == 100)
        backend.resolvePromptLimit = { 5_000 }
        #expect(await backend.effectivePromptLimit() == 5_000)
        backend.resolvePromptLimit = { nil }
        #expect(await backend.effectivePromptLimit() == 100)
    }
}
