echo "push files"
read file
git add $file
git commit -m "update  $file"
git push origin master --force
