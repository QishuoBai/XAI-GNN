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
          v-for="(d, i) in ['train', 'test']"
          :key="d"
          class="d-flex flex-row align-center mt-2 cursor-pointer px-2"
          @click="
            show_dataset[i] = !show_dataset[i];
            add_edge_color_by_filter();
          "
          v-ripple
        >
          <v-icon icon="mdi-check" :size="12" v-if="show_dataset[i]"></v-icon>
          <v-icon icon="mdi-close" :size="12" v-if="!show_dataset[i]"></v-icon>
          <div class="text-caption ml-2">{{ d }}</div>
        </div>
      </div>
      <v-divider class="mb-n1 mt-1"></v-divider>
      <div class="px-2">
        <div
          v-for="(type, i) in selected_types"
          :key="type"
          class="d-flex flex-row align-center mt-2 cursor-pointer px-2"
          @click="
            show_types[i] = !show_types[i];
            add_edge_color_by_filter();
          "
          v-ripple
        >
          <div
            v-if="show_types[i]"
            :style="{
              backgroundColor: type_color[i],
              width: '20px',
              height: '2px',
            }"
          ></div>
          <div
            v-if="!show_types[i]"
            :style="{
              backgroundColor: graph_link_color_bg,
              width: '20px',
              height: '2px',
            }"
          ></div>
          <div class="text-caption ml-2">{{ type }}</div>
        </div>
      </div>
      <v-divider class="mb-n1 mt-1"></v-divider>
      <div class="px-2">
        <div
          class="d-flex flex-row align-center mt-2 cursor-pointer px-2"
          @click="
            show_correct = !show_correct;
            add_edge_color_by_filter();
          "
          v-ripple
        >
          <div
            v-if="show_correct"
            :style="{
              backgroundColor: '#000',
              width: '20px',
              height: '2px',
            }"
          ></div>
          <div
            v-if="!show_correct"
            :style="{
              backgroundColor: graph_link_color_bg,
              width: '20px',
              height: '2px',
            }"
          ></div>
          <div class="text-caption ml-2">correct</div>
        </div>
        <div
          class="d-flex flex-row align-center mt-2 cursor-pointer px-2"
          @click="
            show_wrong = !show_wrong;
            add_edge_color_by_filter();
          "
          v-ripple
        >
          <div v-if="show_wrong" class="d-flex flex-row align-center">
            <div style="width: 4px; height: 2px; background-color: black"></div>
            <div
              style="
                width: 6px;
                height: 2px;
                margin: 0 3px 0 3px;
                background-color: black;
              "
            ></div>
            <div style="width: 4px; height: 2px; background-color: black"></div>
          </div>
          <div v-if="!show_wrong" class="d-flex flex-row align-center">
            <div
              :style="{
                width: '4px',
                height: '2px',
                backgroundColor: graph_link_color_bg,
              }"
            ></div>
            <div
              :style="{
                width: '6px',
                height: '2px',
                margin: '0 3px 0 3px',
                backgroundColor: graph_link_color_bg,
              }"
            ></div>
            <div
              :style="{
                width: '4px',
                height: '2px',
                backgroundColor: graph_link_color_bg,
              }"
            ></div>
          </div>
          <div class="text-caption ml-2">wrong</div>
        </div>
      </div>
    </div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white d-flex flex-column"
      style="top: 10px; right: 10px; max-height: 80%; z-index: 999"
    >
      <div
        class="text-body-2 font-weight-bold d-flex flex-row justify-center align-center"
      >
        Edges Contribution
      </div>
      <v-divider></v-divider>
      <div class="d-flex flex-row align-center text-caption">
        <div
          :style="{ width: panel_cols_layout[0] + 'px' }"
          class="d-flex flex-row align-center justify-center"
        >
          ID
        </div>
        <v-divider vertical></v-divider>
        <div
          :style="{ width: panel_cols_layout[1] + 'px' }"
          class="d-flex flex-row align-center justify-center"
        >
          Label
        </div>
        <v-divider vertical></v-divider>
        <div
          :style="{ width: panel_cols_layout[2] + 'px' }"
          class="d-flex flex-row align-center justify-center"
        >
          Predict
        </div>
        <v-divider vertical></v-divider>
        <div
          :style="{ width: panel_cols_layout[3] + 'px' }"
          class="d-flex flex-row align-center justify-center"
        >
          Contribution Score
        </div>
      </div>
      <v-divider></v-divider>
      <div class="flex-grow-1 overflow-y-auto">
        <div ref="svg_contribution_panel"></div>
      </div>
    </div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white"
      style="bottom: 10px; left: 10px; width: 20%; z-index: 999"
      v-if="show_tooltip_node"
    >
      <div class="text-body-2">Node</div>
      <v-divider></v-divider>
      <div
        class="d-flex flex-row justify-space-between align-center text-caption"
      >
        <div>IP</div>
        <div>{{ tooltip_node.ip }}</div>
      </div>
    </div>
    <div
      class="position-absolute pa-2 rounded elevation-2 bg-white"
      style="bottom: 10px; left: 10px; width: 20%; z-index: 999"
      v-if="show_tooltip_edge"
    >
      <div class="text-body-2">Edge</div>
      <v-divider></v-divider>
      <div
        class="d-flex flex-row justify-space-between align-center text-caption"
      >
        <div>Source IP</div>
        <div>{{ tooltip_edge.src_ip }}</div>
      </div>
      <div
        class="d-flex flex-row justify-space-between align-center text-caption"
      >
        <div>Target IP</div>
        <div>{{ tooltip_edge.dst_ip }}</div>
      </div>
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
      show_correct: true,
      show_wrong: true,
      panel_cols_layout: [50, 70, 70, 140],
      show_tooltip_node: false,
      show_tooltip_edge: false,
      mouseover_enable: true,
      tooltip_node: {
        ip: null,
      },
      tooltip_edge: {
        src_ip: null,
        dst_ip: null,
      },
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
    highlight_edge_id() {
      return globalStore().highlight_edge_id;
    },
    highlight_edge_importance() {
      return globalStore().highlight_edge_importance;
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
    highlight_edge_importance(newVal, oldVal) {
      console.log("highlight_edge_importance", newVal);
      if (newVal !== oldVal && newVal !== null) {
        const data = newVal;
        if (data === null) return [];
        // 清洗数据，将id相同的记录合并并求平均
        const edge_importance = {};
        data.forEach((d) => {
          if (this.all_ids.includes(d.ID)) {
            if (edge_importance[d.ID] === undefined) {
              edge_importance[d.ID] = [];
            }
            edge_importance[d.ID].push(d.importance);
          }
        });
        const edge_importance_avg = {};
        Object.keys(edge_importance).forEach((k) => {
          edge_importance_avg[k] =
            edge_importance[k].reduce((a, b) => a + b) /
            edge_importance[k].length;
        });
        const edge_contribution_list = [];
        Object.keys(edge_importance_avg).forEach((k) => {
          const d = data.find((d) => d.ID == k);
          edge_contribution_list.push({
            ID: d.ID,
            type: types[d.type],
            pred: types[d.pred],
            importance: edge_importance_avg[k],
          });
        });
        //   按照重要性排序
        edge_contribution_list.sort((a, b) => b.importance - a.importance);

        this.draw_edge_importance(edge_importance_avg);
        this.draw_edge_contribution_list(edge_contribution_list);
      } else if (newVal == null) {
        this.clear_edge_importance();
        this.clear_edge_contribution_list();
      }
    },
  },
  methods: {
    // Define your component's methods here
    drawGraph(nodes, links) {
      // Your method logic goes here
      const svg_height = this.$refs.svg_container_main.clientHeight - 10;
      const svg_width = this.$refs.svg_container_main.clientWidth;
      d3.select(this.$refs.svg_container_main).html("");
      this.show_tooltip_edge = false;
      this.show_tooltip_node = false;

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
        .attr("id", "mainview-g-links")
        .selectAll("line")
        .data(links)
        .join("line")
        .attr("class", (d) => {
          const classes = [];
          classes.push("graph-lines");
          classes.push("graph-lines-type-" + types[d.type]);
          classes.push("graph-lines-pred-" + types[d.pred]);
          if (d.type == d.pred) {
            classes.push("graph-lines-correct");
          } else {
            classes.push("graph-lines-wrong");
          }
          if (d.is_train) {
            classes.push("graph-lines-train");
          } else {
            classes.push("graph-lines-test");
          }
          return classes.join(" ");
        })
        .attr("stroke", (d) => {
          const current_type = types[d.type];
          const selected_types = globalStore().selected_types;
          if (selected_types.includes(current_type)) {
            return this.type_color[selected_types.indexOf(current_type)];
          } else {
            return this.graph_link_color_bg;
          }
        })
        .style("cursor", "pointer")
        .attr("stroke-width", 2)
        .attr("stroke-dasharray", (d) => (d.type == d.pred ? "0" : "5,5"))
        .attr("stroke-opacity", 0.7)
        .on("click", (event, d) => {
          globalStore().highlight_edge_id = d.ID;
          globalStore().highlight_edge_src_ip = d.source.ip;
          globalStore().highlight_edge_dst_ip = d.target.ip;
          globalStore().highlight_edge_type = types[d.type];
          globalStore().highlight_edge_pred = types[d.pred];
          this.tooltip_edge.src_ip = d.source.ip;
          this.tooltip_edge.dst_ip = d.target.ip;
          this.show_tooltip_edge = true;
          this.mouseover_enable = false;
        })
        .on("mouseover", (event, d) => {
          if (!this.mouseover_enable) return;
          this.tooltip_edge.src_ip = d.source.ip;
          this.tooltip_edge.dst_ip = d.target.ip;
          this.show_tooltip_edge = true;
        })
        .on("mouseout", (event, d) => {
          if (!this.mouseover_enable) return;
          this.show_tooltip_edge = false;
        });
      // 画node
      const node = g_main
        .append("g")
        .selectAll("circle")
        .data(nodes)
        .join("circle")
        .attr("r", node_radius)
        .attr("fill", this.graph_node_color)
        .style("cursor", "pointer")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5)
        .on("mouseover", (event, d) => {
          if (!this.mouseover_enable) return;
          this.tooltip_node.ip = d.ip;
          this.show_tooltip_node = true;
        })
        .on("mouseout", (event, d) => {
          if (!this.mouseover_enable) return;
          this.show_tooltip_node = false;
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
      svg.on("click", (event) => {
        console.log(event.target.tagName);
        if (event.target.tagName == "svg") {
          this.show_tooltip_node = false;
          this.show_tooltip_edge = false;
          globalStore().highlight_edge_id = null;
          globalStore().highlight_edge_src_ip = null;
          globalStore().highlight_edge_dst_ip = null;
          globalStore().highlight_edge_type = null;
          globalStore().highlight_edge_pred = null;
          this.mouseover_enable = true;
        }
      });
      // Set the position attributes of links and nodes each time the simulation ticks.
      this.simulation.on("tick", () => {
        link
          .attr("x1", (d) => d.source.x)
          .attr("y1", (d) => d.source.y)
          .attr("x2", (d) => d.target.x)
          .attr("y2", (d) => d.target.y);

        node.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
      });
      this.add_edge_color_by_filter();
      //   添加默认点击的点
      const d = links[0];
      globalStore().highlight_edge_id = d.ID;
      globalStore().highlight_edge_src_ip = d.source.ip;
      globalStore().highlight_edge_dst_ip = d.target.ip;
      globalStore().highlight_edge_type = types[d.type];
      globalStore().highlight_edge_pred = types[d.pred];
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

    remove_edge_color_all() {
      d3.selectAll(".graph-lines").attr("stroke", this.graph_link_color_bg);
    },
    add_edge_color_by_filter() {
      d3.selectAll(".graph-lines").attr("stroke", (d) => {
        if (d.is_train) {
          if (!this.show_dataset[0]) {
            return this.graph_link_color_bg;
          }
        } else {
          if (!this.show_dataset[1]) {
            return this.graph_link_color_bg;
          }
        }
        const current_type = types[d.type];
        const selected_types = globalStore().selected_types;
        if (!selected_types.includes(current_type)) {
          return this.graph_link_color_bg;
        }
        if (!this.show_types[selected_types.indexOf(current_type)]) {
          return this.graph_link_color_bg;
        }
        const is_correct = d.type == d.pred;
        if (is_correct) {
          if (!this.show_correct) {
            return this.graph_link_color_bg;
          }
        } else {
          if (!this.show_wrong) {
            return this.graph_link_color_bg;
          }
        }
        return this.type_color[selected_types.indexOf(current_type)];
      });
    },

    // 根据边重要性调整边的样式（粗细等）
    draw_edge_importance(data) {
      // 根据平均重要性调整边的粗细
      const max_width = 10;
      d3.select("#mainview-g-links")
        .selectAll("line")
        .attr("stroke-width", (d) => {
          return data[d.ID] * max_width < 1 ? 1 : data[d.ID] * max_width;
        });
    },
    // 清除边的重要性样式
    clear_edge_importance() {
      d3.select("#mainview-g-links").selectAll("line").attr("stroke-width", 2);
    },
    // 画edge contribution list面板
    draw_edge_contribution_list(data) {
      console.log(data);
      const line_height = 40;
      const svg_height = line_height * data.length;
      const svg_width = this.$refs.svg_contribution_panel.clientWidth;
      d3.select(this.$refs.svg_contribution_panel).html("");
      const panel_cols_layout = this.panel_cols_layout;
      const svg = d3
        .select(this.$refs.svg_contribution_panel)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "visible")
        .attr("width", svg_width)
        .attr("height", svg_height);
      // 画背景
      svg
        .append("rect")
        .attr("width", svg_width)
        .attr("height", svg_height)
        .attr("fill", this.graph_node_color)
        .attr("opacity", 0.1);
      const inner_line_height = 35;
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
        .attr("fill", 'none')
        .attr("stroke", '#000')
        .attr("stroke-width", d => {
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
        .attr("x", panel_cols_layout[0] / 2)
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
        .attr("x", panel_cols_layout[0] + panel_cols_layout[1] / 2)
        .attr("y", (d, i) => i * line_height + line_height / 2)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d) => d.type);
      // 写 pred
      svg
        .append("g")
        .selectAll("text")
        .data(data)
        .join("text")
        .attr("class", "text-caption")
        .attr(
          "x",
          panel_cols_layout[0] + panel_cols_layout[1] + panel_cols_layout[2] / 2
        )
        .attr("y", (d, i) => i * line_height + line_height / 2)
        .attr("text-anchor", "middle")
        .attr("alignment-baseline", "middle")
        .text((d) => d.pred);
      // 画 importance 折线
      // 先画 ticks
      svg
        .append("g")
        .selectAll("line")
        .data(data)
        .join("line")
        .attr(
          "x1",
          panel_cols_layout[0] + panel_cols_layout[1] + panel_cols_layout[2]
        )
        .attr("y1", (d, i) => i * line_height + line_height / 2)
        .attr(
          "x2",
          (d) =>
            panel_cols_layout[0] +
            panel_cols_layout[1] +
            panel_cols_layout[2] +
            panel_cols_layout[3]
        )
        .attr("y2", (d, i) => i * line_height + line_height / 2)
        .attr("stroke", this.graph_node_color)
        .attr("stroke-width", 2);
      // 再画 importance
      svg
        .append("g")
        .selectAll("line")
        .data(data)
        .join("line")
        .attr(
          "x1",
          (d) =>
            panel_cols_layout[0] + panel_cols_layout[1] + panel_cols_layout[2]
        )
        .attr("y1", (d, i) => i * line_height + line_height / 2)
        .attr(
          "x2",
          (d) =>
            panel_cols_layout[0] +
            panel_cols_layout[1] +
            panel_cols_layout[2] +
            panel_cols_layout[3] * d.importance
        )
        .attr("y2", (d, i) => i * line_height + line_height / 2)
        .attr("stroke", (d) => {
          if (globalStore().selected_types.includes(d.type)) {
            return this.type_color[
              globalStore().selected_types.indexOf(d.type)
            ];
          }
        })
        .attr("stroke-width", 5)
        .attr("stroke-linecap", "round");
      // // 画 importance 折线
      // const line = d3.line()
      //     .x((d) => panel_cols_layout[0] + panel_cols_layout[1] + panel_cols_layout[2] + panel_cols_layout[3] * d.importance)
      //     .y((d, i) => i * line_height + line_height / 2);
      // svg
      //     .append("g")
      //     .selectAll("path")
      //     .data(data)
      //     .join("path")
      //     .attr("d", line(data))
      //     .attr("fill", "none")
      //     .attr("stroke", "black")
      //     .attr("stroke-width", 2);
    },
    // 清除edge contribution list面板
    clear_edge_contribution_list() {
      d3.select(this.$refs.svg_contribution_panel).html("");
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
