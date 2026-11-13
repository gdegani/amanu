import Foundation
import Testing

@testable import amanu

/// `amanu doctor`'s summary line: the chain the config actually names, walked
/// against what this machine has — not whether some key or CLI exists
/// somewhere, which is what it used to ask whatever the summary was set to.
struct DoctorSummaryTests {
    private static let pulled = OllamaClient.Model(
        name: "qwen3:8b", bytes: 5_000_000_000, remoteModel: nil, remoteHost: nil)

    private static func check(
        summary: [String: Any], facts: DoctorReport.ModelFacts
    ) -> Check {
        let settings = Config.summary(in: ["summary": summary])
        let route = MeetingEgress.route(
            for: .summary, summary: settings, names: Config.SpeakerNamesSettings())
        return DoctorReport.evaluate("summary", route: route, facts: facts, settings: settings)
    }

    private static func warning(_ check: Check) -> String? {
        if case .warn(let message) = check.status { return message }
        return nil
    }

    private static var everythingButOllama: DoctorReport.ModelFacts {
        var facts = DoctorReport.ModelFacts()
        facts.claudeRuns = true
        facts.codexRuns = true
        facts.anthropicKey = true
        facts.openAIKey = true
        facts.ollama = .unreachable
        return facts
    }

    /// The case the old check got wrong: Ollama chosen, Ollama not running,
    /// and a claude CLI elsewhere on the machine making the line read "ok".
    @Test("Ollama chosen and not running is a warning, whatever else is installed")
    func explicitOllamaNotRunning() throws {
        let check = Self.check(summary: ["backend": "ollama"], facts: Self.everythingButOllama)
        let message = try #require(Self.warning(check))
        #expect(message.contains("nothing answers at http://127.0.0.1:11434"))
        #expect(check.remediation?.contains("start Ollama") == true)
    }

    @Test("Ollama running without the chosen model says which model to pull")
    func explicitOllamaWithoutTheModel() throws {
        var facts = DoctorReport.ModelFacts()
        facts.ollama = .running([OllamaClient.Model(
            name: "llama3:8b", bytes: 1, remoteModel: nil, remoteHost: nil)])
        let check = Self.check(summary: ["backend": "ollama"], facts: facts)
        #expect(Self.warning(check)?.contains("qwen3:8b is not pulled") == true)
        #expect(check.remediation == "ollama pull qwen3:8b")
    }

    @Test("Ollama running with the model, on this Mac, is simply fine")
    func explicitOllamaReady() {
        var facts = DoctorReport.ModelFacts()
        facts.ollama = .running([Self.pulled])
        guard case .ok = Self.check(summary: ["backend": "ollama"], facts: facts).status else {
            Issue.record("A local Ollama with its model is ready.")
            return
        }
    }

    @Test("An Ollama cloud model is ready but said to leave the machine")
    func ollamaCloudModel() {
        var facts = DoctorReport.ModelFacts()
        facts.ollama = .running([OllamaClient.Model(
            name: "qwen3:8b", bytes: 0, remoteModel: "qwen3", remoteHost: "https://ollama.com")])
        #expect(Self.warning(Self.check(summary: ["backend": "ollama"], facts: facts))?
            .contains("leaves this machine") == true)
    }

    @Test("A plain-http Ollama on another machine is refused, and the line says why")
    func lanOllamaOverHTTP() throws {
        let status = DoctorReport.ollamaStatus(baseURL: "http://studio.local:11434")
        guard case .refused(let why) = status else {
            Issue.record("Expected the Base URL to be refused, got \(status)")
            return
        }
        #expect(why.contains("https"))

        var facts = DoctorReport.ModelFacts()
        facts.ollama = status
        let check = Self.check(
            summary: ["backend": "ollama", "ollama_base_url": "http://studio.local:11434"],
            facts: facts)
        #expect(Self.warning(check)?.contains("Base URL is refused") == true)
    }

    @Test("Codex chosen and missing is a warning even with claude installed")
    func explicitCodexMissing() {
        var facts = Self.everythingButOllama
        facts.codexRuns = false
        let check = Self.check(summary: ["backend": "codex-cli"], facts: facts)
        #expect(Self.warning(check)?.contains("codex CLI") == true)
    }

    @Test("auto names the backend that will answer and what follows it")
    func autoNamesTheChain() {
        var facts = DoctorReport.ModelFacts()
        facts.codexRuns = true
        facts.ollama = .running([Self.pulled])
        let message = Self.warning(Self.check(summary: [:], facts: facts))
        #expect(message == "via codex-cli (then ollama) · the transcript leaves this machine")
    }

    @Test("auto with nothing but a local Ollama is fine")
    func autoFallsToLocalOllama() {
        var facts = DoctorReport.ModelFacts()
        facts.ollama = .running([Self.pulled])
        guard case .ok = Self.check(summary: [:], facts: facts).status else {
            Issue.record("Ollama alone on this Mac is a working chain.")
            return
        }
    }

    @Test("auto with nothing ready lists why each backend is not")
    func autoWithNothingReady() throws {
        let message = try #require(Self.warning(
            Self.check(summary: [:], facts: DoctorReport.ModelFacts())))
        #expect(message.hasPrefix("no backend is ready"))
        #expect(message.contains("claude CLI"))
        #expect(message.contains("nothing answers at"))
    }

    @Test("Summaries off, or a backend amanu does not know, are said as such")
    func offAndUnknown() {
        guard case .ok = Self.check(summary: ["enabled": false], facts: .init()).status else {
            Issue.record("Summaries off is not a problem.")
            return
        }
        #expect(Self.warning(Self.check(summary: ["backend": "olama"], facts: .init()))?
            .contains("not one amanu knows") == true)
    }

    @Test("Naming gets a line of its own only when it has a backend of its own",
          .freshHome(config: #"{"summary": {"backend": "none"}}"#))
    func speakerNamesLine() throws {
        #expect(DoctorReport.checkSpeakerNames() == nil)
        try Home.current.writeConfig([
            "summary": ["backend": "none"], "speaker_names": ["backend": "anthropic-api"],
        ])
        let check = try #require(DoctorReport.checkSpeakerNames())
        #expect(check.name == "speaker names")
        #expect(Self.warning(check)?.contains("no Anthropic key") == true)
    }

    // MARK: - LM Studio

    private static func loaded(_ id: String, isLoaded: Bool = true) -> LMStudio.Model {
        LMStudio.Model(
            id: id, type: "llm", isLoaded: isLoaded, loadedContext: isLoaded ? 26_923 : nil,
            maxContext: 262_144)
    }

    @Test("auto does not ask about an LM Studio nobody configured")
    func autoSkipsUnconfiguredLMStudio() {
        let settings = Config.summary(in: nil)
        let route = MeetingEgress.Route(preference: "auto", anthropicModel: nil)
        #expect(!DoctorReport.backends(of: route, settings: settings).contains("lmstudio"))

        let named = Config.summary(in: ["summary": ["lmstudio_model": "m"]])
        let order = DoctorReport.backends(of: route, settings: named)
        #expect(order.suffix(2) == ["lmstudio", "ollama"])
    }

    @Test("LM Studio chosen and not running is a warning that says where it looked")
    func lmstudioNotRunning() throws {
        var facts = Self.everythingButOllama
        facts.lmstudio = .unreachable
        let check = Self.check(
            summary: ["backend": "lmstudio", "lmstudio_model": "m"], facts: facts)
        #expect(Self.warning(check)?.contains("http://127.0.0.1:1234/v1") == true)
        #expect(check.remediation?.contains("start LM Studio") == true)
    }

    @Test("LM Studio without a chosen model names what it has")
    func lmstudioNoModelChosen() throws {
        var facts = DoctorReport.ModelFacts()
        facts.lmstudio = .running([Self.loaded("google/gemma-4-12b")])
        let check = Self.check(summary: ["backend": "lmstudio"], facts: facts)
        #expect(Self.warning(check) == "no LM Studio model is chosen")
        #expect(check.remediation?.contains("google/gemma-4-12b") == true)
    }

    @Test("A model LM Studio lacks is an error, because LM Studio would answer with another")
    func lmstudioMissingModel() throws {
        var facts = DoctorReport.ModelFacts()
        facts.lmstudio = .running([
            Self.loaded("google/gemma-4-12b"),
            LMStudio.Model(id: "embed", type: "embeddings", isLoaded: false,
                           loadedContext: nil, maxContext: nil),
        ])
        for name in ["gemma-4-12b", "embed"] {
            let check = Self.check(
                summary: ["backend": "lmstudio", "lmstudio_model": name], facts: facts)
            #expect(Self.warning(check)?.contains("has no model \(name)") == true, "\(name)")
        }
    }

    @Test("A loaded model on this Mac is simply fine; one that is not loaded is said to be cut small")
    func lmstudioReady() throws {
        var facts = DoctorReport.ModelFacts()
        facts.lmstudio = .running([
            Self.loaded("a"), Self.loaded("b", isLoaded: false),
        ])
        guard case .ok = Self.check(
            summary: ["backend": "lmstudio", "lmstudio_model": "a"], facts: facts).status
        else {
            Issue.record("A loaded model on a local LM Studio is ready.")
            return
        }
        let cold = Self.check(
            summary: ["backend": "lmstudio", "lmstudio_model": "b"], facts: facts)
        #expect(Self.warning(cold)?.contains("not loaded") == true)
        #expect(cold.remediation?.contains("larger context") == true)
    }

    @Test("A plain-http LM Studio on another machine is refused before anybody asks")
    func lanLMStudioOverHTTP() throws {
        let status = DoctorReport.lmstudioStatus(baseURL: "http://studio.local:1234/v1")
        guard case .refused = status else {
            Issue.record("plain http to another machine must be refused, got \(status)")
            return
        }
    }

    @Test("A remote LM Studio over https is ready but said to receive the transcript")
    func remoteLMStudio() throws {
        var facts = DoctorReport.ModelFacts()
        facts.lmstudio = .running([Self.loaded("a")])
        let check = Self.check(
            summary: ["backend": "lmstudio", "lmstudio_model": "a",
                      "lmstudio_base_url": "https://lm.example/v1"], facts: facts)
        #expect(Self.warning(check)?.contains("https://lm.example/v1") == true)
    }
}
