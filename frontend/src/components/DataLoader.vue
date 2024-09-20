<template>
  <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div class="text-body-1 font-weight-bold">Data Loader</div>
    <v-divider></v-divider>
    <div class="d-flex flex-row">
      <div
        :class="
          'flex-grow-1 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
          (tab == 0 ? 'bg-grey' : '')
        "
        v-ripple
        @click="tab = 0"
      >
        Confusion Matrix
      </div>
      <div
        :class="
          'flex-grow-1 text-body-2 d-flex flex-row align-center justify-center cursor-pointer py-1 ' +
          (tab == 1 ? 'bg-grey' : '')
        "
        v-ripple
        @click="tab = 1"
      >
        Cluster Scatters
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data: () => ({
    tab: 0,
  }),
  computed: {},
  watch: {},
  methods: {},
  mounted() {},
};
</script>

<style></style>
