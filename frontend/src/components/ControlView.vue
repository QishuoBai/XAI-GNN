<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between align-center"
    >
      <div>Explainer Controller</div>
      <div>
        <div
          class="text-caption text-uppercase border rounded px-2 py-1 mb-1 elevation-1 cursor-pointer"
          v-ripple
          @click="get_gnnexplainer_result"
        >
          Generate
        </div>
      </div>
    </div>
    <v-divider></v-divider>
    <div class="flex-grow-1 d-flex flex-column w-100">
      <div class="d-flex flex-row">
        <div
          :class="
            'w-50 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
            (tab == 0 ? 'bg-grey' : '')
          "
          v-ripple
          @click="switchTab(0)"
        >
          Feature Importance
        </div>
        <div
          :class="
            'w-50 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
            (tab == 1 ? 'bg-grey' : '')
          "
          v-ripple
          @click="switchTab(1)"
        >
          Topological Importance
        </div>
      </div>
      <v-divider></v-divider>
      <div v-if="tab == 0" class="d-flex flex-column w-100">
        <!-- feat -->
        <div>
          <div
            v-for="(coeff, i) in feat_coeffs"
            :key="coeff.name"
            :class="'d-flex flex-row align-center ' + (i == 0 ? '' : 'mt-n2')"
          >
            <div class="text-caption" style="width: 40%">{{ coeff.label }}</div>
            <v-slider
              v-model="coeff.value"
              :min="i == 0 ? 100 : 0"
              :max="i == 0 ? 2000 : 10"
              :step="i == 0 ? 100 : 0.1"
              :thumb-size="10"
              :track-size="2"
              thumb-label
              hide-details
            />
          </div>
        </div>
      </div>
      <div v-if="tab == 1" class="d-flex flex-column w-100">
        <!-- edge -->
        <div>
          <div
            v-for="(coeff, i) in edge_coeffs"
            :key="coeff.name"
            :class="'d-flex flex-row align-center ' + (i == 0 ? '' : 'mt-n2')"
          >
            <div class="text-caption" style="width: 40%">{{ coeff.label }}</div>
            <v-slider
              v-model="coeff.value"
              :min="i == 0 ? 100 : 0"
              :max="i == 0 ? 2000 : 5"
              :step="i == 0 ? 100 : 0.1"
              :thumb-size="10"
              :track-size="2"
              thumb-label
              hide-details
            />
          </div>
        </div>
      </div>
      <div class="flex-grow-1">
        <div class="h-50 d-flex flex-column">
          <v-divider></v-divider>
          <div class="text-body-2">Original Prediction</div>
          <div class="flex-grow-1" ref="svg_container_original"></div>
        </div>
        <div class="h-50 d-flex flex-column">
          <v-divider></v-divider>
          <div class="text-body-2">
            Prediction with {{ tab == 0 ? "Feature" : "Edge" }} Mask
          </div>
          <div class="flex-grow-1" ref="svg_container_masked"></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { globalStore } from "@/store";
import { postRequest } from "@/utils";
import * as d3 from "d3";
import cm_data from "@/data/cm_ton_iot.json";

const types = cm_data.types;
// Define the component
export default {
  name: "ControlView",
  // Your component's options go here
  data: () => ({
    tab: 0,
    feat_coeffs: [
      { label: "Training Epochs", name: "feat_epochs", value: 10 },
      { label: "Prediction Loss Weight", name: "feat_pred", value: 1.0 },
      { label: "Size Loss Weight", name: "feat_size", value: 1.0 },
      { label: "Entropy Loss Weight", name: "feat_ent", value: 1.0 },
    ],
    edge_coeffs: [
      { label: "Training Epochs", name: "edge_epochs", value: 10 },
      { label: "Prediction Loss Weight", name: "edge_pred", value: 5.0 },
      { label: "Size Loss Weight", name: "edge_size", value: 1.0 },
      { label: "Entropy Loss Weight", name: "edge_ent", value: 1.0 },
    ],
  }),
  computed: {
    // Define the computed properties
    user_coeffs() {
      return {
        feat_epochs: this.feat_coeffs[0].value,
        feat_pred: this.feat_coeffs[1].value,
        feat_size: this.feat_coeffs[2].value,
        feat_ent: this.feat_coeffs[3].value,
        edge_epochs: this.edge_coeffs[0].value,
        edge_pred: this.edge_coeffs[1].value,
        edge_size: this.edge_coeffs[2].value,
        edge_ent: this.edge_coeffs[3].value,
      };
    },
    highlight_edge_id() {
      return globalStore().highlight_edge_id;
    },
  },
  watch: {
    // Define the watch properties
    highlight_edge_id(newVal, oldVal) {
      this.get_gnnexplainer_result();
    },
  },
  methods: {
    switchTab(tab) {
      this.tab = tab;
    },
    get_gnnexplainer_result() {
        // 清除图
      this.clear_prediction();
      globalStore().highlight_feature_importance = null;
      globalStore().highlight_edge_importance = null;

      // 请求后端计算gnnexplainer
      postRequest("/api/get_gnnexplainer_result", {
        ID: this.highlight_edge_id,
        user_coeffs: this.user_coeffs,
      }).then((res) => {
        console.log(res);
        const data = res.data;
        globalStore().highlight_feature_importance = data.feature_importance;
        globalStore().highlight_edge_importance = data.edge_importance;
        this.draw_prediction(
          this.$refs.svg_container_original,
          data.pred_origin_vec
        );
        if (this.tab == 0) {
          this.draw_prediction(
            this.$refs.svg_container_masked,
            data.pred_feature_masked_vec
          );
        } else {
          this.draw_prediction(
            this.$refs.svg_container_masked,
            data.pred_edge_masked_vec
          );
        }
      });
    },
    draw_prediction(container, data) {
      const svg_height = container.clientHeight - 10;
      const svg_width = container.clientWidth;
      d3.select(container).html("");
      const svg = d3
        .select(container)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "hidden")
        .attr("width", svg_width)
        .attr("height", svg_height);
      const margin = { top: 5, right: 15, bottom: 20, left: 15 };
      const innerWidth = svg_width - margin.left - margin.right;
      const innerHeight = svg_height - margin.top - margin.bottom;
      const x = d3
        .scaleBand()
        .domain(types)
        .range([0, innerWidth])
        .padding(0.1);
      const y = d3.scaleLinear().domain([0, 1]).range([innerHeight, 0]);
      const g = svg
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);
      // 画一个灰色背景
      g.append("g")
        .append("rect")
        .attr("x", 0)
        .attr("y", 0)
        .attr("width", innerWidth)
        .attr("height", innerHeight)
        .attr("fill", globalStore().colors.graph_node_color)
        .attr("opacity", 0.1);
      // Draw the x-axis
      g.append("g")
        .call(d3.axisBottom(x))
        .attr("transform", `translate(0,${innerHeight})`)
        .selectAll("text")
        .style("text-anchor", "middle") // 使标签居中
        .attr("transform", "rotate(-20)")
        .style("font-size", "8px");
      // Draw the y-axis
      g.append("g").call(d3.axisLeft(y).ticks(1));

      // Draw the bars
      g.append("g")
        .selectAll("rect")
        .data(data)
        .join("rect")
        .attr("x", (d, i) => x(types[i]))
        .attr("y", (d) => y(d))
        .attr("width", x.bandwidth())
        .attr("height", (d) => innerHeight - y(d))
        .attr("fill", (d, i) => {
          if (globalStore().selected_types.includes(types[i])) {
            return globalStore().colors.type_color[
              globalStore().selected_types.indexOf(types[i])
            ];
          } else {
            return globalStore().colors.graph_node_color;
          }
        });
    },
    clear_prediction() {
      d3.select(this.$refs.svg_container_original).html("");
      d3.select(this.$refs.svg_container_masked).html("");
    },
  },
};
</script>

<style scoped>
/* Your component's styles go here */
</style>
