// G10 oracle: llama.cpp's own chat parser (the code llama-server uses), applied to a
// recorded raw model output.  Built against libllama-common from LLAMA_CPP_DIR.
//
// usage: q35-chatparse-ref CASES.json
//   {"model":"...gguf","cases":[{"name","raw","thinking":bool,"tools":[...]}]}

#include "chat.h"
#include "json.h"
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const char *path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static std::string esc(const std::string &s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20) { char t[8]; snprintf(t, sizeof(t), "\\u%04x", c); o += t; }
            else o += (char)c;
        }
    }
    return o + "\"";
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s CASES.json\n", argv[0]);
        return 2;
    }
    common_json root;
    try {
        root = common_json::parse(read_file(argv[1]));
    } catch (const std::exception &e) {
        fprintf(stderr, "bad JSON: %s\n", e.what());
        return 2;
    }
    const std::string model_path = root.at("model").get<std::string>();
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    llama_model *model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "cannot load vocab from %s\n", model_path.c_str());
        return 1;
    }
    auto tmpls = common_chat_templates_init(model, "", "", "");
    const llama_vocab *vocab = llama_model_get_vocab(model);

    printf("[");
    bool first_case = true;
    for (const auto &c : root.at("cases")) {
        if (!first_case) printf(",");
        first_case = false;
        common_chat_templates_inputs inputs;
        common_chat_msg um;
        um.role = "user";
        um.content = "x";
        inputs.messages = { um };
        inputs.use_jinja = true;
        inputs.enable_thinking = c.value("thinking", false);
        inputs.parallel_tool_calls = c.value("parallel_tool_calls", true);
        inputs.reasoning_format = common_reasoning_format_from_name(
            c.value("reasoning_format", std::string("deepseek")));
        if (c.contains("tools") && !c.at("tools").is_null())
            inputs.tools = common_chat_tools_parse_oaicompat(c.at("tools"));
        common_chat_session session(tmpls.get(), vocab, inputs, {});
        common_chat_msg msg = session.finish(common_chat_input(c.at("raw").get<std::string>()));
        printf("{\"name\":%s,\"content\":%s,\"reasoning_content\":%s,\"tool_calls\":[",
               esc(c.value("name", std::string())).c_str(), esc(msg.content).c_str(),
               esc(msg.reasoning_content).c_str());
        for (size_t i = 0; i < msg.tool_calls.size(); i++) {
            if (i) printf(",");
            printf("{\"name\":%s,\"arguments\":%s}", esc(msg.tool_calls[i].name).c_str(),
                   esc(msg.tool_calls[i].arguments).c_str());
        }
        printf("]}");
    }
    printf("]\n");
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
