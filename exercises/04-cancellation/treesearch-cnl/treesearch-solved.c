#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <omp.h>


#define N 5000000


typedef struct binary_tree_s {
    struct binary_tree_s * left;
    struct binary_tree_s * right;
    int value;
} binary_tree_t;


binary_tree_t * alloc_node() {
    binary_tree_t * node = (binary_tree_t *) malloc(sizeof(binary_tree_t ));
    node->left = node->right = NULL;
    node->value = 0;
    return node;
}


binary_tree_t * insert_value(binary_tree_t * tree, int value) {
    if (!tree) {
        // empty tree, create a new root node
        binary_tree_t * root = alloc_node();
        root ->value = value;
        return root;
    }
    else {
        if (value < tree->value) {
            binary_tree_t * child = insert_value(tree->left, value);
            tree->left = child;
        }
        else {
            binary_tree_t * child = insert_value(tree->right, value);
            tree->right = child;
        }
        return tree;
    }
}


binary_tree_t * create_tree() {
    binary_tree_t * tree = NULL;
    for (size_t i = 0; i < N; i++) {
        tree = insert_value(tree, rand());
    }
    return tree;
}


binary_tree_t * search_tree_sequential_rec(binary_tree_t * tree, int value, int level) {
    binary_tree_t * found = NULL;
    if (tree) {
        if (tree->value == value) {
            found = tree;
        }
        else {
            binary_tree_t * found_left = NULL;
            found_left = search_tree_sequential_rec(tree->left, value, level + 1);
            if (found_left) {
                found = found_left;
                return found;
            }
            binary_tree_t * found_right = NULL;
            found_right = search_tree_sequential_rec(tree->right, value, level + 1);
            if (found_right) {
                found = found_right;
                return found;
            }
        }
    }
    return found;
}


binary_tree_t * search_tree_sequential(binary_tree_t * tree, int value) {
    binary_tree_t * found = NULL;

    found = search_tree_sequential_rec(tree, value, 0);

    return found;
}


binary_tree_t * search_tree_parallel_rec(binary_tree_t * tree, int value, int level) {
    binary_tree_t * found = NULL;
    if (tree) {
        if (tree->value == value) {
            found = tree;
        }
        else {
#pragma omp task shared(found) if(level < 10)
            {
                binary_tree_t * found_left = NULL;
                found_left = search_tree_parallel_rec(tree->left, value, level + 1);
                if (found_left) {
#pragma omp atomic write
                    found = found_left;
                }
            }
#pragma omp task shared(found) if(level < 10)
            {
                binary_tree_t * found_right = NULL;
                found_right = search_tree_parallel_rec(tree->right, value, level + 1);
                if (found_right) {
#pragma omp atomic write
                    found = found_right;
                }
            }
#pragma omp taskwait
        }
    }
    return found;
}


binary_tree_t * search_tree_parallel(binary_tree_t * tree, int value) {
    binary_tree_t * found = NULL;

#pragma omp parallel shared(found,tree,value)
    {
#pragma omp master
        {
            found = search_tree_parallel_rec(tree, value, 0);
        }
    }

    return found;
}


binary_tree_t * search_tree_rec_cancellation(binary_tree_t * tree, int value, int level) {
    binary_tree_t * found = NULL;
    if (tree) {
        if (tree->value == value) {
            found = tree;
        }
        else {
#pragma omp task shared(found) if(level < 10)
            {
                binary_tree_t * found_left = NULL;
                found_left = search_tree_rec_cancellation(tree->left, value, level + 1);
                if (found_left) {
#pragma omp atomic write
                    found = found_left;
#pragma omp cancel taskgroup
                }
            }
#pragma omp task shared(found) if(level < 10)
            {
                binary_tree_t * found_right = NULL;
                found_right = search_tree_rec_cancellation(tree->right, value, level + 1);
                if (found_right) {
#pragma omp atomic write
                    found = found_right;
#pragma omp cancel taskgroup
                }
            }
#pragma omp taskwait
        }
    }
    return found;
}


binary_tree_t * search_tree_parallel_cancellation(binary_tree_t * tree, int value) {
    binary_tree_t * found = NULL;

#pragma omp parallel shared(found,tree,value)
    {
#pragma omp master
        {
#pragma omp taskgroup
            {
                found = search_tree_rec_cancellation(tree, value, 0);
            }
        }
    }

    return found;
}

#if DUMP_TREE
void dump_tree(binary_tree_t  *tree, bool called_for_root) {
    if (!tree) {
        printf("()");
    }
    else {
        printf(" (%d ", tree->value);
        dump_tree(tree->left, false);
        dump_tree(tree->right, false);
        printf(")");
    }
    if (called_for_root) {
        printf("\n");
    }
}
#endif


#if DUMP_TREE_IN_ORDER
void dump_tree_in_order(binary_tree_t * tree, bool called_for_root) {
    if (tree) {
        dump_tree_in_order(tree->left, false);
        printf("%d ", tree->value);
        dump_tree_in_order(tree->right, false);
    }
    if (called_for_root) {
        printf("\n");
    }
}
#endif


int main(int argc, char const * argv[]) {
    binary_tree_t * tree = NULL;
    binary_tree_t * found = NULL;
    double start,end;
    double seq,par,cncl;

    /* construct a tree */
    fprintf(stdout, "creating tree... ");
    fflush(stdout);
    tree = create_tree();
    fprintf(stdout, "done\n");
#if DUMP_TREE
    dump_tree(tree, true);
#endif
#if DUMP_TREE_IN_ORDER
    dump_tree_in_order(tree, true);
#endif

    /* warm-up */
    int search_for = 2146969233;
    fprintf(stdout, "warm-up... ");
    fflush(stdout);
    found = search_tree_sequential(tree, search_for);
    fprintf(stdout, "done\n");

    /* sequential search */
    fprintf(stdout, "sequential search... ");
    fflush(stdout);
    start = omp_get_wtime();
    found = search_tree_sequential(tree, search_for);
    end = omp_get_wtime();
    seq = end - start;
    if (found) {
        fprintf(stdout, "found %d\n", found->value);
    }
    else {
        fprintf(stdout, "value %d not found\n", search_for);
    }

    /* parallel search */
    fprintf(stdout, "parallel search... ");
    fflush(stdout);
    start = omp_get_wtime();
    found = search_tree_parallel(tree, search_for);
    end = omp_get_wtime();
    par = end - start;
    if (found) {
        fprintf(stdout, "found %d\n", found->value);
    }
    else {
        fprintf(stdout, "value %d not found\n", search_for);
    }

    /* parallel search with cancellation */
    fprintf(stdout, "parallel search w/ cancellation... ");
    fflush(stdout);
    start = omp_get_wtime();
    found = search_tree_parallel_cancellation(tree, search_for);
    end = omp_get_wtime();
    cncl = end - start;
    if (found) {
        fprintf(stdout, "found %d\n", found->value);
    }
    else {
        fprintf(stdout, "value %d not found\n", search_for);
    }

    fprintf(stdout, "timing: seq=%lf par=%lf cancellation=%lf\n", seq, par, cncl);

    return 0;
}
