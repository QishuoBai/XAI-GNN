<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between"
    >
      <div>Data Loader</div>
      <div class="d-flex flex-row align-center">
        <v-dialog v-model="info_dialog" max-width="600">
          <template v-slot:activator="{ props: activatorProps }">
            <v-icon
              icon="mdi-information"
              :size="18"
              v-bind="activatorProps"
            ></v-icon>
          </template>

          <v-card>
            <v-card-title class="text-body-1">
              <v-icon icon="mdi-information" size="small"></v-icon>
              Dataset Information
            </v-card-title>
            <v-card-text>
              <div>
                <div
                  class="d-flex flex-row align-center text-body-2 font-weight-bold"
                >
                  <div>Basic</div>
                  <v-divider class="ml-2"></v-divider>
                </div>
                <v-container class="pa-0 text-body-2 mt-2">
                  <v-row no-gutters>
                    <v-col cols="3">Name</v-col>
                    <v-col cols="9">TON_IoT Dataset</v-col>
                  </v-row>
                  <v-row no-gutters class="mt-2">
                    <v-col cols="3">Description</v-col>
                    <v-col cols="9"
                      >The TON_IoT datasets are new generations of Internet of
                      Things (IoT) and Industrial IoT (IIoT) datasets for
                      evaluating the fidelity and efficiency of different
                      cybersecurity applications based on Artificial
                      Intelligence (AI).
                    </v-col>
                  </v-row>
                </v-container>
                <div
                  class="d-flex flex-row align-center text-body-2 font-weight-bold"
                >
                  <div>Statistical</div>
                  <v-divider class="ml-2"></v-divider>
                </div>
                <v-container class="pa-0 text-body-2 mt-2">
                  <v-row no-gutters>
                    <v-col cols="3">Number of Nodes</v-col>
                    <v-col cols="9">{{ dataset_description.nodes_num }}</v-col>
                  </v-row>
                  <v-row no-gutters class="mt-2">
                    <v-col cols="3">Number of Edges</v-col>
                    <v-col cols="9">{{ dataset_description.links_num }} </v-col>
                  </v-row>
                  <v-row no-gutters class="mt-2">
                    <v-col cols="3">Number of Types</v-col>
                    <v-col cols="9"
                      >{{ dataset_description.type_num.length }}
                    </v-col>
                  </v-row>
                </v-container>
                <div
                  class="d-flex flex-row align-center text-body-2 font-weight-bold mt-2"
                >
                  <div>Proportion</div>
                  <v-divider class="ml-2"></v-divider>
                </div>
                <v-container class="pa-0 text-body-2 mt-2">
                  <v-row
                    v-for="(item, i) in dataset_description.type_num"
                    :key="i"
                    no-gutters
                    class="mt-2"
                  >
                    <v-col cols="3">{{ item.type }}</v-col>
                    <v-col cols="9">
                      <div class="d-flex flex-row align-center text-caption">
                        <div
                          :style="{
                            height: '20px',
                            width: (item.num / 300000) * 300 + 'px',
                            backgroundColor: dataset_desc_bar_color,
                          }"
                        ></div>
                        <div class="ml-2">
                          - {{ item.num }} ({{
                            (
                              (item.num / dataset_description.links_num) *
                              100
                            ).toFixed(2)
                          }}%)
                        </div>
                      </div>
                    </v-col>
                  </v-row>
                </v-container>
              </div>
            </v-card-text>
          </v-card>
        </v-dialog>
        <v-dialog v-model="config_dialog" max-width="600">
          <template v-slot:activator="{ props: activatorProps }">
            <v-icon
              icon="mdi-wrench"
              size="x-small"
              class="ml-2"
              v-bind="activatorProps"
            ></v-icon>
          </template>

          <v-card>
            <v-card-title class="text-body-1">
              <v-icon icon="mdi-wrench" size="small"></v-icon>
              Select 5 Types to Analyze
            </v-card-title>
            <v-card-text>
              <v-select
                chips
                label="Types"
                :items="types"
                multiple
                density="comfortable"
                variant="outlined"
                v-model="selected_types"
                clearable
                :rules="config_rules"
              ></v-select
            ></v-card-text>
            <v-card-actions>
              <v-spacer></v-spacer>
              <v-btn
                text="Confirm"
                variant="text"
                @click="confirmConfig()"
              ></v-btn>
            </v-card-actions>
          </v-card>
        </v-dialog>
      </div>
    </div>
    <v-divider></v-divider>
    <div class="flex-grow-1 d-flex flex-column w-100">
      <div class="d-flex flex-row">
        <div
          :class="
            'flex-grow-1 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
            (tab == 0 ? 'bg-grey' : '')
          "
          v-ripple
          @click="switchTab(0)"
        >
          Confusion Matrix
        </div>
        <div
          :class="
            'flex-grow-1 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
            (tab == 1 ? 'bg-grey' : '')
          "
          v-ripple
          @click="switchTab(1)"
        >
          Cluster Scatters
        </div>
      </div>
      <v-divider></v-divider>
      <div v-if="tab == 0" class="flex-grow-1 d-flex flex-column w-100">
        <!-- cm -->
        <div ref="svg_container_0" class="flex-grow-1"></div>
        <div>
          <div class="d-flex flex-row align-center">
            <div class="text-caption" style="width: 30%">Number of Edges</div>
            <v-slider
              v-model="tab_config0_edge_num_limit"
              :min="0"
              :max="max_edges_num"
              :step="1"
              :thumb-size="10"
              :track-size="2"
              thumb-label
              hide-details
            />
          </div>
          <div class="d-flex flex-row align-center mt-n2">
            <div class="text-caption" style="width: 30%">Target ID Range</div>
            <v-range-slider
              v-model="tab_config0_edge_id_range"
              :min="0"
              :max="tab_config0_edge_id_range_max"
              :step="1"
              :thumb-size="10"
              :track-size="2"
              thumb-label
              hide-details
            />
          </div>
        </div>
      </div>
      <div v-if="tab == 1" class="flex-grow-1">
        <!-- scatter -->
      </div>
      <div
        class="text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 bg-grey rounded"
        @click="loadData()"
        v-ripple
      >
        Load Data
      </div>
    </div>
  </div>
</template>

<script>
import cm_data from "@/data/cm_ton_iot.json";
import edgelist from "@/data/edgelist_ton_iot.json";
import { globalStore } from "@/store";
import * as d3 from "d3";
import { postRequest, getRequest } from "@/utils";
import dataset_description from "@/data/dataset_description.json";

const types = cm_data.types;
const max_edges_num = 500;

export default {
  data: () => ({
    tab: 0,
    config_dialog: false,
    info_dialog: false,
    types: types,
    selected_types: globalStore().selected_types,
    config_rules: [(v) => v.length == 5 || "Please select 5 types"],
    max_edges_num: max_edges_num,
    tab_config0_selected_cell: [0, 0],
    tab_config0_edge_num_limit: max_edges_num / 2,
    tab_config0_edge_id_range: [0, 0],
    tab_config0_edge_id_range_max: 1,
    tab_config1_edge_num_limit: max_edges_num / 2,
    tab_config1_edge_id_range: [0, 0],
    tab_config1_edge_id_range_max: 1,
    dataset_description: dataset_description,
    dataset_desc_bar_color: globalStore().colors.dataset_desc_bar_color,
  }),
  computed: {},
  watch: {
    tab_config0_edge_id_range(newVal, oldVal) {
      const [newMin, newMax] = newVal;
      const [oldMin, oldMax] = oldVal;
      if (newMax == oldMax) {
        // 变小
        if (newMax - newMin > this.tab_config0_edge_num_limit) {
          this.tab_config0_edge_id_range = [
            newMin,
            newMin + this.tab_config0_edge_num_limit,
          ];
        }
      } else {
        // 变大
        if (newMax - newMin > this.tab_config0_edge_num_limit) {
          this.tab_config0_edge_id_range = [
            newMax - this.tab_config0_edge_num_limit,
            newMax,
          ];
        }
      }
    },
    tab_config0_edge_num_limit(val) {
      const [min, max] = this.tab_config0_edge_id_range;
      if (max - min > val) {
        this.tab_config0_edge_id_range = [min, min + val];
      }
    },
  },
  methods: {
    confirmConfig() {
      if (this.selected_types.length != 5) return;
      this.config_dialog = false;
      globalStore().selected_types = [...this.selected_types];
      this.draw_cm(this.selected_types);
    },
    draw_cm(selected_types) {
      const svg_height = this.$refs.svg_container_0.clientHeight - 10;
      const svg_width = this.$refs.svg_container_0.clientWidth;
      d3.select(this.$refs.svg_container_0).html("");
      const svg = d3
        .select(this.$refs.svg_container_0)
        .append("svg")
        .attr("viewBox", `0 0 ${svg_width} ${svg_height}`)
        .attr("overflow", "visible")
        .attr("width", svg_width)
        .attr("height", svg_height);
      const selected_types_index = selected_types.map((type) =>
        types.indexOf(type)
      );
      const matrix = [];
      for (let i = 0; i < selected_types_index.length; i++) {
        matrix.push([]);
        for (let j = 0; j < selected_types_index.length; j++) {
          matrix[i].push(
            cm_data.cm[selected_types_index[i]][selected_types_index[j]]
          );
        }
      }
      const main_width = svg_width * 0.7;
      const main_height = svg_height * 0.85;
      const text_left_width = svg_width - main_width;
      const text_top_height = svg_height - main_height;
      const rect_padding = 6;
      let rect_width =
        (main_width - (selected_types_index.length - 1) * rect_padding) /
        selected_types_index.length;
      let rect_height =
        (main_height - (selected_types_index.length - 1) * rect_padding) /
        selected_types_index.length;
      rect_width = rect_width > rect_height ? rect_height : rect_width;
      rect_height = rect_width;
      // 画cell
      let g_cells = svg
        .append("g")
        .attr("transform", `translate(${text_left_width}, ${text_top_height})`);
      const opacity_scale = d3
        .scaleLinear()
        .domain([
          d3.min(matrix.flat()),
          d3.median(matrix.flat()),
          d3.max(matrix.flat()),
        ])
        .range([0.1, 0.6, 0.9]);
      for (let i = 0; i < selected_types_index.length; i++) {
        for (let j = 0; j < selected_types_index.length; j++) {
          let current_g = g_cells.append("g").style("cursor", "pointer");
          current_g
            .append("rect")
            .attr("x", j * (rect_width + rect_padding))
            .attr("y", i * (rect_height + rect_padding))
            .attr("class", "rect-cm-cells rect-cm-cells-" + i + "-" + j)
            .attr("width", rect_width)
            .attr("height", rect_height)
            .attr("fill", globalStore().colors.cm_color)
            .attr("rx", 5)
            .attr("ry", 5)
            .attr("fill-opacity", opacity_scale(matrix[i][j]))
            .attr("stroke", "black")
            .attr("stroke-width", 0);
          current_g
            .append("text")
            .attr("x", j * (rect_width + rect_padding) + rect_width / 2)
            .attr("y", i * (rect_height + rect_padding) + rect_height / 2)
            .attr("text-anchor", "middle")
            .attr("dominant-baseline", "middle")
            .style("font-size", "0.75rem")
            .text(matrix[i][j]);
          current_g.on("click", () => {
            this.select_cell(matrix, i, j);
          });
        }
      }
      // 画text
      let g_text_top = svg.append("g");
      let g_text_left = svg.append("g");
      for (let i = 0; i < selected_types_index.length; i++) {
        g_text_top
          .append("text")
          .attr(
            "x",
            text_left_width + i * (rect_width + rect_padding) + rect_width / 2
          )
          .attr("y", text_top_height - 10)
          .attr("text-anchor", "middle")
          .attr("dominant-baseline", "middle")
          .style("font-size", "0.75rem")
          .text(selected_types[i]);
        g_text_left
          .append("text")
          .attr("x", text_left_width - 10)
          .attr(
            "y",
            text_top_height + i * (rect_height + rect_padding) + rect_height / 2
          )
          .attr("text-anchor", "end")
          .attr("dominant-baseline", "middle")
          .style("font-size", "0.75rem")
          .text(selected_types[i]);
      }
      let g_text_top_title = svg.append("g");
      let g_text_left_title = svg.append("g");
      g_text_top_title
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "0.75rem")
        .attr(
          "transform",
          `translate(${
            text_left_width + d3.min([main_width, main_height]) / 2
          }, 15)`
        )
        .text("Predict");
      g_text_left_title
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", "0.75rem")
        .attr(
          "transform",
          `translate(20, ${
            text_top_height + d3.min([main_width, main_height]) / 2
          }) rotate(-90)`
        )

        .text("Label");
      // 画边框
      this.select_cell(
        matrix,
        this.tab_config0_selected_cell[0],
        this.tab_config0_selected_cell[1]
      );
    },
    select_cell(matrix, i, j) {
      this.tab_config0_selected_cell = [i, j];
      console.log(i, j);
      d3.selectAll(".rect-cm-cells").attr("stroke-width", 0);
      d3.select(".rect-cm-cells-" + i + "-" + j).attr("stroke-width", 2);
      // 修改 taget id range
      this.tab_config0_edge_id_range_max = matrix[i][j];
      this.tab_config0_edge_id_range = [
        0,
        d3.min([
          this.tab_config0_edge_id_range_max,
          this.tab_config0_edge_num_limit,
        ]),
      ];
    },
    switchTab(tab) {
      this.tab = tab;
      if (tab == 0) {
        this.$nextTick(() => {
          this.draw_cm(globalStore().selected_types);
        });
      }
    },
    loadData() {
      if (this.tab == 0) {
        const selected_types = globalStore().selected_types;
        let request_data = {
          row_type: selected_types[this.tab_config0_selected_cell[0]],
          col_type: selected_types[this.tab_config0_selected_cell[1]],
          edge_num: this.tab_config0_edge_num_limit,
          edge_id_range: this.tab_config0_edge_id_range,
        };
        postRequest("/api/data_loader/cm", request_data).then((res) => {
          console.log(res.data);
          const all_ids = res.data.all_ids;
          const target_ids = res.data.target_ids;
          const nodes = res.data.nodes;
          const links = res.data.links;
          globalStore().all_ids = [...all_ids];
          globalStore().target_ids = [...target_ids];
          globalStore().nodes = [...nodes];
          globalStore().links = [...links];
        });
      }
    },
  },
  mounted() {
    if (this.tab == 0) {
      this.draw_cm(this.selected_types);
      this.loadData();
    }
  },
};
</script>

<style></style>
