<template>
  <div class="d-flex flex-column h-screen w-screen bg-grey-lighten-2">
    <div
      id="header"
      class="border-b-thin pa-1 px-3 bg-grey-darken-4 d-flex flex-row justify-space-between align-center"
    >
      <div class="text-h6 d-flex flex-row align-center">
        <div class="ml-0">GNNExplorer</div>
      </div>
      <div class="text-body-2 text-grey-lighten-1 d-flex flex-row align-center">
        <div class="ml-0">
          A visualization approach for exploring and analyzing Graph Neural
          Networks (GNNs)
        </div>
        <v-dialog v-model="about_dialog" max-width="600">
          <template v-slot:activator="{ props: activatorProps }">
            <div
              class="d-flex flex-row align-center ml-4 cursor-pointer"
              v-bind="activatorProps"
            >
              <v-icon icon="mdi-help-circle" size="x-small"></v-icon>
              <div>About</div>
            </div>
          </template>
          <v-card>
            <v-card-title class="text-body-1">
              <v-icon icon="mdi-help-circle" size="small"></v-icon>
              About
            </v-card-title>
            <v-card-text class="text-body-2">
                <p>
                    <span class="font-weight-bold">GNNExplorer</span> is a tool designed to help users visualize and analyze Graph Neural Networks (GNNs). 
                    It provides various views and functionalities to load data, explore the main graph structure, 
                    inspect features, and review the history of interactions.
                </p>
                <p class="mt-2">
                    This tool aims to make it easier for researchers and practitioners to understand the behavior 
                    and performance of GNNs through interactive visualizations.
                </p>
            </v-card-text>
          </v-card>
        </v-dialog>
      </div>
    </div>
    <div class="flex-grow-1 d-flex flex-row w-100" style="height: 0px">
      <div style="width: 20%" class="ma-1 d-flex flex-column">
        <div style="height: 50%" class="bg-white rounded mb-1">
          <DataLoader />
        </div>
        <div class="flex-grow-1 bg-white rounded"></div>
      </div>
      <div class="flex-grow-1 ma-1 ml-0 d-flex flex-column">
        <div style="height: 70%" class="bg-white rounded mb-1">
          <MainView />
        </div>
        <div class="flex-grow-1 bg-white rounded"><FeatureView /></div>
      </div>
      <div style="width: 20%" class="ma-1 ml-0 d-flex flex-column">
        <div style="height: 40%" class="bg-white rounded mb-1">
          <HistoryView />
        </div>
        <div class="flex-grow-1 bg-white rounded">
          <InfoView />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data: () => ({
    about_dialog: false,
  }),
  computed: {},
  watch: {},
  methods: {},
  mounted() {},
};
</script>

<style>
* {
  font-family: "Roboto";
}
</style>
