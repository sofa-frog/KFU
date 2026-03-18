# Lab 1: Introduction to PyTorch Tensor Operations

**Course:** Neural Networks and Their Applications  
**Student:** Bichurina S.P., Group 09-261, 3rd year  
**University:** Kazan (Volga Region) Federal University, Institute of Computational Mathematics and Information Technologies, Department of Information Systems

---

## Task Description

The goal of this lab was to explore and demonstrate basic PyTorch tensor manipulation functions. Each function was tested with practical examples to understand its behavior and use cases.

## Functions Explored

### 1. torch.cat()

This function concatenates multiple tensors along a specified dimension. Concatenation along axis 0 increases the number of rows in the resulting tensor, while concatenation along axis 1 appends columns, effectively "gluing" the second tensor to the right of the first one.

### 2. torch.gather()

Gathers values from a tensor based on specified indices along a given dimension. The result is a tensor whose shape matches the index tensor, where each element is selected from the input tensor according to the corresponding index.

### 3. torch.masked_select()

Given a tensor and a boolean mask of the same shape, this function extracts elements from the tensor where the mask value is `True`.

### 4. torch.narrow()

Extracts a specific slice from a tensor — a sub-array of a given size starting from a particular index along a specified dimension. For example, it can return rows from the second to the third from a matrix.

### 5. torch.nonzero()

Returns the indices of all non-zero elements in a tensor.

### 6. torch.split()

Splits a tensor into several parts of a given size along a specified dimension. For instance, a tensor can be split into chunks of 2 elements each.

### 7. torch.squeeze()

Removes all dimensions of size 1 from a tensor, simplifying its shape for further processing.

### 8. torch.unsqueeze()

Adds a new dimension of size 1 at a specified position in the tensor — the inverse of `squeeze()`.

### 9. torch.unbind()

Splits a tensor along a specified dimension into a tuple of individual tensors (one per slice along that dimension).

### 10. torch.where()

Returns a tensor containing values from one input tensor where a condition is true, and values from another tensor where the condition is false.
