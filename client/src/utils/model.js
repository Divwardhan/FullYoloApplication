export function getModelImageUrl(model) {
  // Use model.image or model.url or fallback to a placeholder
  console.log("********************************",model);
  return model.image || model.url ||'https://placehold.co/96x96?text=YOLO';
} 