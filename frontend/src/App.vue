<template>
  <div class="d-flex flex-column h-screen w-screen bg-grey-lighten-2">
    <div
      id="header"
      class="border-b-thin pa-1 px-3 bg-grey-darken-4 d-flex flex-row justify-space-between align-center"
    >
      <div class="text-h6 d-flex flex-row align-center">
        <!-- <img src="/logo_white.png" style="height: 30px" /> -->
        <div class="ml-0">GNNExplorer</div>
      </div>
      <div class="text-body-2 text-grey-lighten-1 d-flex flex-row align-center">
        <!-- <img src="/logo_white.png" style="height: 30px" /> -->
        <div class="ml-0">
          A visualization approach for exploring and analyzing Graph Neural
          Networks (GNNs)
        </div>
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
  data: () => ({}),
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
