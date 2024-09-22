import { link } from "d3";
import { defineStore } from "pinia";

export const globalStore = defineStore("cluster", {
    state: () => {
        return {
            selected_types: ["normal","backdoor","ddos","dos", "injection"],
            
            colors:{
                cm_color: "#85182a",
                graph_node_color: "#cccccc",
                graph_link_color_bg: "#cccccc7f",
                type_color: ['#8b1e3f', '#15616d', '#9fcc2e', '#ff7d00', '#78290f'],
            },
            all_ids: [],
            target_ids: [],
            nodes: [],
            links: [],
        };
    },
});
