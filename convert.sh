# Process all common image files, maintaining aspect ratio with padding
for img in *.jpg *.jpeg *.png *.gif; do
  # Skip if no matching files
  [ -e "$img" ] || continue
  
  # Resize the image, maintain aspect ratio, add white padding if needed
  convert "$img" -resize 768x1024 -background white -gravity center -extent 768x1024 "$img"
  echo "Processed: $img"
done
