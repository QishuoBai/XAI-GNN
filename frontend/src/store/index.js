import { defineStore } from "pinia";

export const globalStore = defineStore("cluster", {
    state: () => {
        return {
            selected_types: ["normal","backdoor","ddos","dos", "injection"],
            colors:{
                cm_color: "#85182a"
            },
            all_ids: [],
            target_ids: [],
        };
    },
});
