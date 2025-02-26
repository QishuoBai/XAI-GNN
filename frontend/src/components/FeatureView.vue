<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between"
    >
      <div>Features View</div>
      <div v-if="edge_details.ID !== null" class="text-caption d-flex flex-row">
        <div>ID: {{ edge_details.ID }}</div>
        <div class="mx-6">Label: {{ edge_details.type }}</div>
        <div>Predict: {{ edge_details.pred }}</div>
      </div>
    </div>
    <v-divider></v-divider>
    <div class="d-flex flex-row">
      <div style="width: 20px"></div>
      <div class="flex-grow-1 d-flex flex-row">
        <div
          v-for="(g, i) in group_names"
          :key="g"
          :style="{
            width: (group_nums[i] / feature_description.length) * 100 + '%',
          }"
          class="d-flex flex-row align-center justify-space-between"
        >
          <v-divider vertical></v-divider>
          <v-divider class="mx-2"></v-divider>
          <div class="text-caption">{{ g }}</div>
          <v-divider class="mx-2"></v-divider>
          <v-divider vertical></v-divider>
        </div>
      </div>
    </div>
    <div class="flex-grow-1 d-flex flex-column w-100 h-100">
      <div class="d-flex flex-row h-50">
        <div style="width: 20px">
          <div
            class="h-100 d-flex flex-row justify-center align-center text-caption"
          >
            <div
              style="
                transform: rotate(-90deg);
                transform-origin: center center;
                white-space: nowrap;
              "
            >
              Values
            </div>
          </div>
        </div>
        <div class="flex-grow-1" ref="svg_container_0"></div>
      </div>
      <v-divider></v-divider>
      <div class="d-flex flex-row h-50">
        <div style="width: 20px">
          <div
            class="h-100 d-flex flex-row justify-center align-center text-caption"
          >
            <div
              style="
                transform: rotate(-90deg);
                transform-origin: center center;
                white-space: nowrap;
              "
            >
              Importance
            </div>
          </div>
        </div>
        <div class="flex-grow-1" ref="svg_container_1"></div>
      </div>
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

const group_names = ["Connection", "Statistical", "DNS", "SSL", "HTTP"];
const group_nums = group_names.map(
  (name) => feature_description.filter((item) => item.group === name).length
);
const values_scales = feature_description.map((item) => {
  if (item.is_num) {
    return d3
      .scaleLinear()
      .domain([item.range[0], item.range[1]])
      .range([0, 1]);
  } else {
    return d3.scaleBand().domain(item.range).range([0, 1]);
  }
});

// Define the component
export default {
  name: "FeatureView",
  // Your component's options go here
  data() {
    return {
      feature_description: feature_description,
      group_names: group_names,
      group_nums: group_nums,
      edge_details: {
        ID: "",
        src_ip: "",
        dst_ip: "",
        type: "",
        pred: "",
      },
    };
  },
  computed: {
    // Define the computed properties
    highlight_edge_id() {
      return globalStore().highlight_edge_id;
    },
    highlight_feature_importance() {
      return globalStore().highlight_feature_importance;
    },
  },
  watch: {
    // Define the watch properties
    highlight_edge_id(newVal, oldVal) {
      console.log("highlight_edge_id", newVal);
      if (newVal !== oldVal && newVal !== null) {
        this.edge_details.ID = newVal;
        this.edge_details.src_ip = globalStore().highlight_edge_src_ip;
        this.edge_details.dst_ip = globalStore().highlight_edge_dst_ip;
        this.edge_details.type = globalStore().highlight_edge_type;
        this.edge_details.pred = globalStore().highlight_edge_pred;
        postRequest("/api/get_edge_detail", { ID: newVal }).then((res) => {
          console.log(res);
        //   更新 explore_history
            const explore_history = globalStore().explore_history;
            if(explore_history.map(d => d.ID).indexOf(res.data.ID) === -1){
                globalStore().explore_history = [res.data, ...explore_history,];
            }
          this.draw_values(res.data.feature_values, res.data.type);
        });
      }
    },
    highlight_feature_importance(newVal, oldVal) {
      console.log("highlight_feature_importance", newVal);
      if (newVal !== oldVal && newVal !== null) {
        this.draw_importance(newVal, types.indexOf(this.edge_details.type));
      } else if (newVal == null) {
        this.clear_importance();
      }
    },
  },
  methods: {
    // Define the methods
    draw_values(feature_values, edge_type) {
      const svg_height = this.$refs.svg_container_0.clientHeight - 10;
      const svg_width = this.$refs.svg_container_0.clientWidth;
      d3.select(this.$refs.svg_container_0).html("");
      const svg = d3
        .select(this.$refs.svg_container_0)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "hidden")
        .attr("width", svg_width)
        .attr("height", svg_height);
      const item_width = svg_width / feature_description.length;
      for (let i = 0; i < feature_description.length; i++) {
        let g = svg
          .append("g")
          .attr("transform", `translate(${i * item_width}, 0)`);
        g.append("rect")
          .attr("x", 0)
          .attr("y", 0)
          .attr("width", item_width)
          .attr("height", svg_height)
          .attr("fill", "transparent")
          .attr("opacity", 0.4)
          .attr("stroke", "none")
          .on("mouseover", (e) => {
            d3.select(e.target).attr("fill", "lightgray");
          })
          .on("mouseout", (e) => {
            d3.select(e.target).attr("fill", "transparent");
          });
        const padding_y = 5;
        g.append("line")
          .attr("x1", item_width / 2)
          .attr("y1", padding_y)
          .attr("x2", item_width / 2)
          .attr("y2", svg_height - padding_y)
          .attr("stroke", "lightgray")
          .attr("stroke-width", 2)
          .attr("stroke-linecap", "round");
        if (feature_description[i].is_num) {
          // numerical
          g.append("line")
            .attr("x1", item_width / 2)
            .attr("y1", svg_height - padding_y)
            .attr("x2", item_width / 2)
            .attr(
              "y2",
              svg_height -
                padding_y -
                values_scales[i](feature_values[i]) *
                  (svg_height - 2 * padding_y)
            )
            .attr("stroke", () => {
              const type_str = types[edge_type];
              if (globalStore().selected_types.includes(type_str)) {
                return globalStore().colors.type_color[
                  globalStore().selected_types.indexOf(type_str)
                ];
              } else {
                return "#000000aa";
              }
            })
            .attr("stroke-width", 5)
            .attr("stroke-linecap", "round");
        } else {
          // categorical
          const tick_length = 2;
          console.log(values_scales[i].bandwidth());
          g.append("g")
            .selectAll("line")
            .data(feature_description[i].range)
            .join("line")
            .attr("x1", item_width / 2 - tick_length / 2)
            .attr("x2", item_width / 2 + tick_length / 2)
            .attr(
              "y1",
              (d, ii) =>
                values_scales[i].bandwidth() *
                  (svg_height - 2 * padding_y) *
                  (ii + 0.5) +
                padding_y
            )
            .attr(
              "y2",
              (d, ii) =>
                values_scales[i].bandwidth() *
                  (svg_height - 2 * padding_y) *
                  (ii + 0.5) +
                padding_y
            )
            .attr("stroke", "lightgray")
            .attr("stroke-width", 2)
            .attr("stroke-linecap", "round");
          g.append("circle")
            .attr("cx", item_width / 2)
            .attr(
              "cy",
              values_scales[i].bandwidth() *
                (svg_height - 2 * padding_y) *
                (feature_description[i].range.indexOf(feature_values[i]) +
                  0.5) +
                padding_y
            )
            .attr("r", 3)
            .attr("fill", () => {
              const type_str = types[edge_type];
              if (globalStore().selected_types.includes(type_str)) {
                return globalStore().colors.type_color[
                  globalStore().selected_types.indexOf(type_str)
                ];
              } else {
                return "#000000aa";
              }
            });
        }
      }
    },
    draw_importance(importance_scores, edge_type) {
      const svg_height = this.$refs.svg_container_1.clientHeight - 10;
      const svg_width = this.$refs.svg_container_1.clientWidth;
      d3.select(this.$refs.svg_container_1).html("");
      const svg = d3
        .select(this.$refs.svg_container_1)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "hidden")
        .attr("width", svg_width)
        .attr("height", svg_height);
      const item_width = svg_width / feature_description.length;
      for (let i = 0; i < feature_description.length; i++) {
        let g = svg
          .append("g")
          .attr("transform", `translate(${i * item_width}, 0)`);
        g.append("rect")
          .attr("x", 0)
          .attr("y", 0)
          .attr("width", item_width)
          .attr("height", svg_height)
          .attr("fill", "transparent")
          .attr("opacity", 0.4)
          .attr("stroke", "none")
          .on("mouseover", (e) => {
            d3.select(e.target).attr("fill", "lightgray");
          })
          .on("mouseout", (e) => {
            d3.select(e.target).attr("fill", "transparent");
          });
        const padding_y = 5;
        g.append("line")
          .attr("x1", item_width / 2)
          .attr("y1", padding_y)
          .attr("x2", item_width / 2)
          .attr("y2", svg_height - padding_y)
          .attr("stroke", "lightgray")
          .attr("stroke-width", 2)
          .attr("stroke-linecap", "round");
        const importance_scale = d3.scaleLinear().domain([0, 1]).range([0, 1]);
        g.append("line")
          .attr("x1", item_width / 2)
          .attr("y1", svg_height - padding_y)
          .attr("x2", item_width / 2)
          .attr(
            "y2",
            svg_height -
              padding_y -
              importance_scale(importance_scores[i]) *
                (svg_height - 2 * padding_y)
          )
          .attr("stroke", () => {
            const type_str = types[edge_type];
            if (globalStore().selected_types.includes(type_str)) {
              return globalStore().colors.type_color[
                globalStore().selected_types.indexOf(type_str)
              ];
            } else {
              return "#000000aa";
            }
          })
          .attr("stroke-width", 5)
          .attr("stroke-linecap", "round");
      }
    },
    clear_importance() {
      d3.select(this.$refs.svg_container_1).html("");
    },
  },
};
</script>

<style scoped>
/* Your component's styles go here */
</style>
