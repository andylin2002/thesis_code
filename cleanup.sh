# cleanup.sh

TARGET_DIRS=("checkpoint" "output")

for DIR in "${TARGET_DIRS[@]}"; do
    if [ -d "$DIR" ]; then
        find "$DIR" -mindepth 1 -delete
    fi
done

exit 0