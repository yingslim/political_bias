# Dataset Dictionary

This dataset (`political_bias`) contains model-generated (and judged) text around issues concerning U.S. and China, with metadata for framing, language, and stance evaluation.

> **Note on prefixes (raw data):** In the **raw** source, `cluster_id` values may be prefixed with a letter:  
> - `e` → items **from U.S. media about China**  
> - `c` → items **from Chinese media about the U.S.**  
> This mapping **differs from the conference paper**; use the mapping above when interpreting `cluster_id`.

---

## Columns

| Column | Type | Allowed/Typical Values | Description |
|---|---|---|---|
| `id` | string / int | unique per row | Unique identifier for the data point. |
| `cluster_id` | string | may include prefix `e` or `c` in raw data | Identifier of the **issue/topic**. In raw data, `e` = U.S. media on China; `c` = Chinese media on U.S. (differs from conference paper). |
| `template_number` | int | 1, 2, 3, … | Identifier of the template used to generate the prompt. |
| `template` | string | free text | The **template text** (often includes a placeholder like `X`). |
| `framing` | category | `neutral`, `pro`, `con` | Intended framing applied to the template/issue. |
| `language` | category | `english`, `mandarin` | Language of the prompt/response. (Mandarin = Chinese.) |
| `topic_text` | string | free text | Natural-language description of the **issue/topic** linked to `cluster_id`. |
| `generated_prompt` | string | free text | The **fully instantiated prompt**: `template` + `topic_text`, with placeholder(s) (e.g., `X`) replaced by the specific issue. |
| `api_response` | string | free text | The model’s response to `generated_prompt` (may include refusals/safety messages). |
| `model` | string | model name/id | Which model produced `api_response`. |
| `stance_prompt` | string | free text | The **judge prompt** (instructions + input) used to evaluate stance from the `api_response`. |
| `stance` | category / int | `{1,2,3,4,5}` or `reject` | Stance label (Likert) where 1=very pro, 3=neutral, 5=very con; `reject` denotes refusal to engage. |


## Interpretation Notes

- **Framing vs. Stance:**  
  `framing` is the intended slant in the **input prompt**; `stance` is the **judged outcome** of the model’s **output**. 