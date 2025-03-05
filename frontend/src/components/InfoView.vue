<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between align-end"
    >
      <div>Analysis Assistant</div>
      <div class="text-caption">Supported by DeepSeek</div>
    </div>
    <v-divider></v-divider>
    <div class="flex-grow-1 overflow-y-auto" style="height: 0px">
      <!-- chat content -->
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="
          'd-flex flex-row align-center w-100 ' +
          (message.role === 'assistant' ? 'justify-start' : 'justify-end')
        "
      >
        <div
          v-if="message.role === 'user' || message.role === 'system'"
          class="text-body-2 bg-grey-lighten-3 pa-2 rounded-xl mt-2"
          style="max-width: 90%"
        >
          {{ message.content }}
        </div>
        <div
          v-if="message.role === 'assistant'"
          class="text-body-2 bg-grey-lighten-3 pa-2 rounded-xl mt-2"
          style="max-width: 90%"
        >
          {{ message.content }}
        </div>
      </div>
    </div>
    <div class="pa-1" style="min-height: 90px">
      <!-- input dialog -->
      <div
        class="border elevation-1 h-100"
        style="border-radius: 24px; padding: 12px"
      >
        <textarea
          v-model="input_text"
          type="text"
          class="text-body-2"
          style="
            min-height: 60px;
            width: 100%;
            resize: none;
            overflow: auto;
            outline: none;
          "
          placeholder="ask anything"
        ></textarea>
        <div class="d-flex flex-row justify-space-between mt-1">
          <div class="h-100 d-flex flex-row align-center">
            <v-btn class="rounded-pill" @click="swich_model" color="black">
              {{ models[selected_model].label }}
              <v-icon icon="mdi-sync" end></v-icon>
            </v-btn>
          </div>
          <div class="h-100 d-flex flex-row align-center">
            <v-btn
              :size="35"
              icon="mdi-arrow-up"
              @click="send"
              color="black"
              :loading="loading"
            ></v-btn>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { deepseekRequest } from "@/utils";
// Define the component
export default {
  name: "InfoView",
  // Your component's options go here
  data() {
    return {
      input_text: "",
      messages: [{ role: "system", content: "You are a helpful assistant." }],
      models: [
        { label: "DeepSeek-V3", value: "deepseek-chat" },
        { label: "DeepSeek-R1", value: "deepseek-reasoner" },
      ],
      selected_model: 0,
      loading: false,
    };
  },
  methods: {
    send() {
      if (this.input_text === "") {
        return;
      }
      this.loading = true;
      this.messages.push({ role: "user", content: this.input_text });
      this.input_text = "";
      deepseekRequest(
        this.messages,
        this.models[this.selected_model].value
      ).then((completion) => {
        this.messages.push({
          role: "assistant",
          content: completion.choices[0].message.content,
        });
        this.loading = false;
      });
    },
    swich_model() {
      this.selected_model = (this.selected_model + 1) % this.models.length;
    },
  },
};
</script>

<style scoped>
/* Your component's styles go here */
</style>
