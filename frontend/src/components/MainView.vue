<template>
  <div
    class="h-100 w-100 pa-2 d-flex flex-row justify-center align-center position-relative"
  >
    <!-- Your component's HTML template goes here -->
    <div class="h-100 w-100 position-relative" ref="svg_container_main"></div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white"
      style="top: 10px; left: 10px; z-index: 999"
    >
      <div
        class="text-body-2 font-weight-bold d-flex flex-row justify-center align-center px-2"
      >
        Filter Panel
      </div>
      <v-divider class="mb-n1"></v-divider>
      <div class="px-2">
        <div
          v-for="(type, i) in selected_types"
          :key="type"
          class="d-flex flex-row align-center my-2 cursor-pointer px-2"
          @click="show_types[i] = !show_types[i]"
          v-ripple
        >
          <div
            v-if="show_types[i]"
            class="rounded-circle"
            :style="{
              backgroundColor: type_color[i],
              width: '10px',
              height: '10px',
            }"
          ></div>
          <div
            v-if="!show_types[i]"
            class="rounded-circle"
            :style="{
              backgroundColor: graph_link_color_bg,
              width: '10px',
              height: '10px',
            }"
          ></div>
          <div class="text-caption ml-2">{{ type }}</div>
        </div>
      </div>
      <v-divider class="my-n1"></v-divider>
      <div class="px-2">
        <div
          v-for="(d, i) in ['train', 'test']"
          :key="d"
          class="d-flex flex-row align-center mt-2 cursor-pointer px-2"
          @click="show_dataset[i] = !show_dataset[i]"
          v-ripple
        >
          <div
            v-if="show_dataset[i]"
            class="rounded-circle"
            :style="{
              backgroundColor: '#000',
              width: '10px',
              height: '10px',
            }"
          ></div>
          <div
            v-if="!show_dataset[i]"
            class="rounded-circle"
            :style="{
              backgroundColor: graph_link_color_bg,
              width: '10px',
              height: '10px',
            }"
          ></div>
          <div class="text-caption ml-2">{{ d }}</div>
        </div>
      </div>
    </div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white"
      style="top: 10px; right: 10px; height: 80%; width: 20%; z-index: 999"
    >
      <div
        class="text-body-2 font-weight-bold d-flex flex-row justify-center align-center"
      >
        Recommendation Panel
      </div>
      <v-divider></v-divider>
      <!-- <div class="d-flex flex-row align-center text-body-2">
        <div class="d-flex flex-row align-center justify-center">ID</div>
      </div> -->
      <v-container class="pa-0">
        <v-row no-gutters class="text-caption">
          <v-col
            :cols="recommendation_cols_layout[0]"
            class="d-flex flex-row justify-center align-center"
            >ID</v-col
          >
          <v-divider vertical></v-divider>
          <v-col
            :cols="recommendation_cols_layout[1]"
            class="d-flex flex-row justify-center align-center"
            >Label</v-col
          >
          <v-divider vertical></v-divider>
          <v-col
            :cols="recommendation_cols_layout[2]"
            class="d-flex flex-row justify-center align-center"
            >Predict</v-col
          >
          <v-divider vertical></v-divider>
          <v-col
            :cols="recommendation_cols_layout[3]"
            class="d-flex flex-row justify-center align-center"
            >Details</v-col
          >
        </v-row>
        <v-divider></v-divider>
      </v-container>
    </div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white"
      style="bottom: 10px; left: 10px; height: 30%; width: 20%; z-index: 999"
      v-if="true"
    >
      Tooltip
    </div>
  </div>
</template>

<script>
// Import any necessary dependencies here
import { globalStore } from "@/store";
import cm_data from "@/data/cm_ton_iot.json";
import * as d3 from "d3";

const types = cm_data.types;

const node_radius = 5;
const link_distance = 70;
const link_strength = 0.5;
const collide_radius = 10;

export default {
  name: "MainView",
  // Your component's script logic goes here
  data() {
    return {
      // Define your component's data properties here
      simulation: null,
      show_types: [true, true, true, true, true],
      show_dataset: [true, true],
      recommendation_cols_layout: [3, 3, 3, 3],
      show_tooltip: false,
    };
  },
  computed: {
    // Define computed properties here
    all_ids() {
      return globalStore().all_ids;
    },
    target_ids() {
      return globalStore().target_ids;
    },
    nodes() {
      return globalStore().nodes;
    },
    links() {
      return globalStore().links;
    },
    graph_node_color() {
      return globalStore().colors.graph_node_color;
    },
    graph_link_color_bg() {
      return globalStore().colors.graph_link_color_bg;
    },
    type_color() {
      return globalStore().colors.type_color;
    },
    selected_types() {
      return globalStore().selected_types;
    },
  },
  watch: {
    // Define watch properties here
    all_ids() {
      this.$nextTick(() => {
        console.log("draw new graph");
        console.log(globalStore().nodes);
        console.log(globalStore().links);
        this.drawGraph(globalStore().nodes, globalStore().links);
      });
    },
  },
  methods: {
    // Define your component's methods here
    drawGraph(nodes, links) {
      // Your method logic goes here
      const svg_height = this.$refs.svg_container_main.clientHeight - 10;
      const svg_width = this.$refs.svg_container_main.clientWidth;
      d3.select(this.$refs.svg_container_main).html("");
      const svg = d3
        .select(this.$refs.svg_container_main)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "hidden")
        .attr("width", svg_width)
        .attr("height", svg_height);
      let g_main = svg.append("g");

      // Create a simulation with several forces.
      if (this.simulation) {
        this.simulation.stop();
      }
      this.simulation = d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink(links)
            .id((d) => d.ip)
            .distance(link_distance)
            .strength(link_strength)
        )
        .force("charge", d3.forceManyBody())
        .force("collide", d3.forceCollide().radius(collide_radius))
        .force("x", d3.forceX(svg_width / 2))
        .force("y", d3.forceY(svg_height / 2));
      //   画link
      const link = g_main
        .append("g")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("stroke", (d) => {
          const current_type = types[d.type];
          const selected_types = globalStore().selected_types;
          if (selected_types.includes(current_type)) {
            return this.type_color[selected_types.indexOf(current_type)];
          } else {
            return this.graph_link_color_bg;
          }
        })
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", (d) => (d.type == d.pred ? "0" : "5,5"))
        .attr("stroke-opacity", 0.7)
        .on("mouseover", (event, d) => {
          this.show_tooltip = true;
        })
        .on("mouseout", (event, d) => {
          this.show_tooltip = false;
        });
      // 画node
      const node = g_main
        .append("g")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", node_radius)
        .attr("fill", this.graph_node_color)
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .on("mouseover", (event, d) => {
          this.show_tooltip = true;
        })
        .on("mouseout", (event, d) => {
          this.show_tooltip = false;
        });
      // Add a drag behavior.
      node.call(
        d3
          .drag()
          .on("start", this.dragstarted)
          .on("drag", this.dragged)
          .on("end", this.dragended)
      );
      // 添加缩放功能
      const zoom = d3
        .zoom()
        .scaleExtent([0.5, 10]) // 设置缩放的范围，0.5倍到10倍
        .on("zoom", (event) => {
          g_main.attr("transform", event.transform); // 缩放和移动g_main
        });

      // 将缩放绑定到svg
      svg.call(zoom);
      // Set the position attributes of links and nodes each time the simulation ticks.
      this.simulation.on("tick", () => {
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);

        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      });
    },
    // Reheat the simulation when drag starts, and fix the subject position.
    dragstarted(event) {
      if (!event.active) this.simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    },

    // Update the subject (dragged node) position during drag.
    dragged(event) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    },

    // Restore the target alpha so the simulation cools after dragging ends.
    // Unfix the subject position now that it’s no longer being dragged.
    dragended(event) {
      if (!event.active) this.simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    },
  },
  mounted() {
    // Your component's mounted hook goes here
  },
};
</script>

<style scoped>
/* Your component's styles go here */
</style>
