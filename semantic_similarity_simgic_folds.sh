set -e

for fold in 0 1 2 3 4 5 6 7 8 9
do
    echo "  Fold $fold"
    groovy semantic_similarity_simgic.groovy -fold $fold
    done
done

