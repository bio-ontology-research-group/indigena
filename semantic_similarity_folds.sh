set -e

# Define pairs of (pw, gw) values to iterate over
pw_gw_pairs=(
    # "resnik bma"
    "resnik bmm"
    "lin bma"
    "lin bmm"

)

# Iterate over each pair
for pair in "${pw_gw_pairs[@]}"
do
    # Split the pair into pw and gw
    read -r pw gw <<< "$pair"
    
    echo "Running with pw=$pw, gw=$gw"
    
    # Run for all folds with this pair
    for fold in 0 1 2 3 4 5 6 7 8 9
    do
        echo "  Fold $fold"
        groovy semantic_similarity.groovy -pw $pw -gw $gw -fold $fold
    done
done

