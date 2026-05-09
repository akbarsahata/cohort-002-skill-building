You've built a powerful search pipeline that combines keyword matching and semantic search. Now it's time to put that pipeline to work by creating a tool that your AI agent can actually call.

This transforms your search system from a standalone feature into an integral part of your agent's decision-making process. Your agent will be able to autonomously search through emails to find the information it needs to answer questions.

We'll create the tool, integrate it into your agent, and then refine it with prompt engineering to get the best results.

## Adding Search Tool to Agent

<!-- VIDEO -->

Let's implement a search tool that the AI agent can use to answer questions by searching through emails.

Recommendation: hand-code this commit, paying particular attention to the prompting. This is new ground we've not covered in skill-building.

### Steps To Complete

#### Creating the search tool file

- [ ] Create a new file `src/app/api/chat/search-tool.ts` with the basic imports:

```typescript
// src/app/api/chat/search-tool.ts
import {
  loadEmails,
  reciprocalRankFusion,
  searchWithBM25,
  searchWithEmbeddings,
} from '@/app/search';
import { tool } from 'ai';
import { z } from 'zod';
```

#### Defining the tool description and input schema

- [ ] Create the `searchTool` with a description and input schema that defines `keywords` and `searchQuery` as optional parameters:

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
export const searchTool = tool({
  description:
    'Search emails using both keyword and semantic search. Returns most relevant emails ranked by reciprocal rank fusion.',
  inputSchema: z.object({
    keywords: z
      .array(z.string())
      .describe(
        'Exact keywords for BM25 search (names, amounts, specific terms)',
      )
      .optional(),
    searchQuery: z
      .string()
      .describe(
        'Natural language query for semantic search (broader concepts)',
      )
      .optional(),
  }),
});
```

</Spoiler>

#### Implementing the execute function - loading and searching

- [ ] Add the `execute` function that loads emails and performs both BM25 and embedding-based searches:

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
export const searchTool = tool({
  description:
    'Search emails using both keyword and semantic search. Returns most relevant emails ranked by reciprocal rank fusion.',
  inputSchema: z.object({
    keywords: z
      .array(z.string())
      .describe(
        'Exact keywords for BM25 search (names, amounts, specific terms)',
      )
      .optional(),
    searchQuery: z
      .string()
      .describe(
        'Natural language query for semantic search (broader concepts)',
      )
      .optional(),
  }),
  // ADDED: Execute function to run the search
  execute: async ({ keywords, searchQuery }) => {
    console.log('Keywords:', keywords);
    console.log('Search query:', searchQuery);

    const emails = await loadEmails();

    // ADDED: Perform BM25 and embedding searches
    const bm25Results = keywords
      ? await searchWithBM25(keywords, emails)
      : [];
    const embeddingResults = searchQuery
      ? await searchWithEmbeddings(searchQuery, emails)
      : [];
    // ADDED: Combine results using reciprocal rank fusion
    const rrfResults = reciprocalRankFusion([
      bm25Results.slice(0, 30),
      embeddingResults.slice(0, 30),
    ]);
  },
});
```

</Spoiler>

#### Implementing the execute function - returning results

- [ ] Complete the `execute` function by filtering and formatting the top 10 results:

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
execute: async ({ keywords, searchQuery }) => {
  console.log('Keywords:', keywords);
  console.log('Search query:', searchQuery);

  const emails = await loadEmails();

  const bm25Results = keywords
    ? await searchWithBM25(keywords, emails)
    : [];
  const embeddingResults = searchQuery
    ? await searchWithEmbeddings(searchQuery, emails)
    : [];
  const rrfResults = reciprocalRankFusion([
    bm25Results.slice(0, 30),
    embeddingResults.slice(0, 30),
  ]);

  // ADDED: Filter and map top results
  const topEmails = rrfResults
    .slice(0, 10)
    .filter((r) => r.score > 0)
    .map((r) => ({
      id: r.email.id,
      from: r.email.from,
      to: r.email.to,
      subject: r.email.subject,
      body: r.email.body,
      timestamp: r.email.timestamp,
      score: r.score,
    }));

  return {
    emails: topEmails,
  };
},
```

</Spoiler>

#### Updating the chat route

- [ ] Import the `searchTool` and `stepCountIs` in `src/app/api/chat/route.ts`:

```typescript
// src/app/api/chat/route.ts
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  safeValidateUIMessages,
  stepCountIs,
  streamText,
  UIMessage,
} from 'ai';
// ADDED: Import the search tool
import { searchTool } from './search-tool';
```

- [ ] Update the `streamText` configuration to use the search tool and `gemini-2.5-flash`:

<Spoiler>

```typescript
// src/app/api/chat/route.ts
const result = streamText({
  // CHANGED: Updated model from gemini-2.5-flash-lite
  model: google('gemini-2.5-flash'),
  messages: convertToModelMessages(messages),
  // ADDED: Add system prompt directing agent to use search
  system: `
    Use your search tool to answer questions.
  `,
  // ADDED: Add the search tool to available tools
  tools: {
    search: searchTool,
  },
  // ADDED: Stop after 10 steps to prevent infinite loops
  stopWhen: [stepCountIs(10)],
});
```

</Spoiler>

#### Removing debug logs

- [ ] Clean up the `searchWithEmbeddings` function in `src/app/search.ts` by removing the debug log:

<Spoiler>

```typescript
// src/app/search.ts
export async function searchWithEmbeddings(
  query: string,
  emails: Email[],
) {
  const emailEmbeddings = await loadOrGenerateEmbeddings(emails);

  const { embedding: queryEmbedding } = await embed({
    model: google.textEmbeddingModel('gemini-embedding-001'),
    value: query,
  });

  const results = emailEmbeddings.map(({ id, embedding }) => {
    const email = emails.find((e) => e.id === id)!;
    const score = cosineSimilarity(queryEmbedding, embedding);
    return { score, email };
  });

  // DELETED: Removed console.log("Results:", results.length);
  return results.sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Testing the search tool

- [ ] Run your application and test by asking the AI agent a question:

```bash
pnpm dev
```

- [ ] Try asking "What are Sarah's hobbies?" in the chat.

The agent will use the search tool to find relevant emails and provide an answer based on the results.

---

## Context Engineering Search Tool

<!-- VIDEO -->

Let's improve the system prompt for your email assistant. Better structure and clear anti-hallucination guardrails will help the agent use the search tool effectively.

Recommendation: hand-code this commit. More new ground.

### Steps To Complete

#### Restructuring the system prompt

- [ ] Replace the simple system prompt with a structured prompt including context, rules, and instructions:

<Spoiler>

```typescript
// src/app/api/chat/route.ts
const result = streamText({
  model: google('gemini-2.5-flash'),
  messages: convertToModelMessages(messages),
  // CHANGED: Restructured system prompt with clear sections
  system: `
<task-context>
You are an email assistant that helps users find and understand information from their emails.
</task-context>

<rules>
- You MUST use the search tool for ANY question about emails, people, amounts, dates, or specific information
- NEVER answer from your training data - always search the actual emails first
- If the first search doesn't find enough information, try different keywords or search queries
- Use both semantic (searchQuery) and keyword (keywords) search parameters together for best results
- Only after searching should you formulate your answer based on the search results
</rules>

<the-ask>
Here is the user's question. Search their emails first, then provide your answer based on what you find.
</the-ask>
  `,
  tools: {
    search: searchTool,
  },
  stopWhen: [stepCountIs(10)],
});
```

</Spoiler>

#### Testing the changes

- [ ] Test the agent with a query like "what house did Sarah buy?":

```bash
pnpm dev
```

- [ ] Verify the agent searches the emails and provides accurate results based on what it finds.

The improved prompt structure enables the agent to work reliably while preventing hallucinations by enforcing tool usage.
