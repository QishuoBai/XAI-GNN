<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between align-center"
    >
      <div>Explore History</div>
      <div>
        <div
          class="text-caption text-uppercase border rounded px-2 py-1 mb-1 elevation-1 cursor-pointer"
          v-ripple
          @click="clear_history"
        >
          clear
        </div>
      </div>
    </div>
    <v-divider></v-divider>
    <div class="text-caption d-flex flex-row w-100">
      <div
        v-for="item in cols_layout"
        :key="item.text"
        class="d-flex flex-row align-center justify-center"
        :style="{ width: (item.width / cols_layout_sum) * 100 + '%' }"
      >
        {{ item.text }}
      </div>
    </div>
    <div class="flex-grow-1 overflow-y-auto">
      <div ref="svg_container"></div>
    </div>
  </div>
</template>

<script>
import { globalStore } from "@/store";
import { postRequest } from "@/utils";
import feature_description from "@/data/feature_description.json";
import * as d3 from "d3";
import cm_data from "@/data/cm_ton_iot.json";

const types = cm_data.types;
// Define the component
export default {
  name: "HistoryView",
  // Your component's options go here
  data() {
    return {
      cols_layout: [
        { text: "ID", width: 1 },
        { text: "Label", width: 1 },
        { text: "Prediction", width: 1 },
        { text: "Details", width: 2 },
      ],
    };
  },
  computed: {
    cols_layout_sum() {
      return d3.sum(this.cols_layout.map((d) => d.width));
    },
    explore_history() {
      return globalStore().explore_history;
    },
  },
  watch: {
    explore_history(newVal) {
      this.draw_history(newVal);
    },
  },
  methods: {
    clear_history() {
      globalStore().explore_history = [];
    },
    draw_history(data) {
      const line_height = 50;
      const inner_line_height = 40;
      const svg_height = line_height * data.length;
      const svg_width = this.$refs.svg_container.clientWidth;
      d3.select(this.$refs.svg_container).html("");
      const svg = d3
        .select(this.$refs.svg_container)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "hidden")
        .attr("width", svg_width)
        .attr("height", svg_height);

      // 画背景
      svg
        .append("g")
        .selectAll("rect")
        .data(data)
        .join("rect")
        .attr("x", 0)
        .attr(
          "y",
          (d, i) => i * line_height + (line_height - inner_line_height) / 2
        )
        .attr("rx", 5)
        .attr("ry", 5)
        .attr("width", svg_width)
        .attr("height", inner_line_height)
        .attr("fill", (d, i) => (i % 2 === 0 ? "#f0f0f0" : "#f1f1f1"))
        .attr("stroke", "#000")
        .attr("stroke-width", (d) => {
          if (globalStore().highlight_edge_id == d.ID) {
            return 1;
          } else {
            return 0;
          }
        });
      // 写IDs
      svg
        .append("g")
        .selectAll("text")
        .data(data)
        .join("text")
        .attr("class", "text-caption")
        .attr(
          "x",
          ((this.cols_layout[0].width / this.cols_layout_sum) * svg_width) / 2
        )
        .attr("y", (d, i) => i * line_height + line_height / 2)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d) => d.ID);
      // 写 types
      svg
        .append("g")
        .selectAll("text")
        .data(data)
        .join("text")
        .attr("class", "text-caption")
        .attr(
          "x",
          ((this.cols_layout[0].width + this.cols_layout[1].width / 2) /
            this.cols_layout_sum) *
            svg_width
        )
        .attr("y", (d, i) => i * line_height + line_height / 2)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d) => types[d.type]);
      // 写 pred
      svg
        .append("g")
        .selectAll("text")
        .data(data)
        .join("text")
        .attr("class", "text-caption")
        .attr(
          "x",
          ((this.cols_layout[0].width +
            this.cols_layout[1].width +
            this.cols_layout[2].width / 2) /
            this.cols_layout_sum) *
            svg_width
        )
        .attr("y", (d, i) => i * line_height + line_height / 2)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d) => types[d.pred]);
      // 根据pred_detail 画 details 的柱状图
      const x = d3
        .scaleBand()
        .domain(types)
        .range([
          0,
          (this.cols_layout[3].width / this.cols_layout_sum) * svg_width,
        ]);
      const y = d3.scaleLinear().domain([0, 1]).range([0, inner_line_height]);
      svg
        .append("g")
        .selectAll("g")
        .data(data)
        .join("g")
        .attr(
          "transform",
          (d, i) =>
            `translate(${
              ((this.cols_layout[0].width +
                this.cols_layout[1].width +
                this.cols_layout[2].width) /
                this.cols_layout_sum) *
              svg_width
            }, ${i * line_height + (line_height - inner_line_height) / 2})`
        )
        .selectAll("rect")
        .data((d) => d.pred_detail)
        .join("rect")
        .attr("x", (d, i) => x(types[i]))
        .attr("y", (d) => inner_line_height - y(d))
        .attr("width", x.bandwidth())
        .attr("height", (d) => y(d))
        .attr("fill", (d, i) =>
          globalStore().selected_types.includes(types[i])
            ? globalStore().colors.type_color[
                globalStore().selected_types.indexOf(types[i])
              ]
            : globalStore().colors.graph_node_color
        );
    },
  },
};
</script>

<style scoped>
/* Your component's styles go here */
</style>
