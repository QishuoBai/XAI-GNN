<template>
    <div class="h-100 w-100 pa-2 d-flex flex-column">
    <div
      class="text-body-1 font-weight-bold d-flex flex-row justify-space-between align-end"
    >
      <div>Analysis Assistant</div>
      <div class="text-caption">Powered by LLM</div>
    </div>
    <v-divider></v-divider>
    <div class="flex-grow-1 overflow-y-auto" style="height: 0px;">
        <div class="bg-grey-lighten-1" style="height: 1000px;">
            <!-- chat content -->
        </div>
    </div>
    <div class="bg-grey-lighten-5 rounded" style="min-height: 70px;">
            <!-- input dialog -->
        </div>
  </div>
</template>

<script>

// Define the component
export default {
    name: 'InfoView',
    // Your component's options go here
};
</script>

<style scoped>
/* Your component's styles go here */
</style>