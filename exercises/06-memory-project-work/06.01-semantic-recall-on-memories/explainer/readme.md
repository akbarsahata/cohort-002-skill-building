We have a memory system that stores information about users, but right now it's just a static database. When a user interacts with our agent, we're not actually _using_ those memories to inform the conversation.

This is where semantic recall comes in. We need to intelligently pull relevant memories from our data bank based on what the user is saying, and inject them into the agent's context.

By the end of this section, you'll have implemented both semantic memory retrieval and memory extraction - so the agent can both access existing memories and create new ones as it learns about users.

## Making Embeddings Functions Generic

<!-- VIDEO -->

Throughout this section, we'll be embedding different types of items. Making search functions generic allows them to work with any type, not just email chunks. This commit converts the search functions to accept generic types with a `toText` conversion function.

Recommendation: cherry-pick if you're not that interested in generic types in TypeScript. But worth hand-coding if you are.

### Steps To Complete

#### Making `searchWithBM25` generic

- [ ] Update `searchWithBM25` to accept a generic type `T` instead of `EmailChunk[]`. Add a `toText` function parameter to convert items to strings. This involves adding a generic type parameter and a callback function to the function signature.

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: Made function generic with type parameter T
export async function searchWithBM25<T>(
  keywords: string[],
  items: T[], // CHANGED: Generic items instead of emailChunks
  toText: (item: T) => string, // ADDED: Conversion function
) {
  // CHANGED: Use toText function instead of hardcoded emailChunkToText
  const corpus = items.flatMap((item) => toText(item));

  const scores: number[] = (BM25 as any)(corpus, keywords);

  // CHANGED: Return generic item instead of email
  return scores
    .map((score, idx) => ({ score, item: items[idx] }))
    .sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Making `loadOrGenerateEmbeddings` generic

- [ ] Update `loadOrGenerateEmbeddings` to accept a generic type `T` and a `toText` function. This allows embedding any type of item, not just email chunks.

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: Made function generic with type parameter T
export async function loadOrGenerateEmbeddings<T>(
  items: T[], // CHANGED: Generic items instead of emailChunks
  toText: (item: T) => string, // ADDED: Conversion function
): Promise<{ item: T; embedding: number[] }[]> {
  await ensureEmbeddingsCacheDirectory();

  // CHANGED: Generic result type
  const results: { item: T; embedding: number[] }[] = [];
  const uncachedItems: T[] = [];

  // CHANGED: Use toText function
  for (const item of items) {
    const cachedEmbedding = await getCachedEmbedding(
      toText(item),
    );
    if (cachedEmbedding) {
      results.push({ item, embedding: cachedEmbedding });
    } else {
      uncachedItems.push(item);
    }
  }

  if (uncachedItems.length > 0) {
    console.log(
      `Generating embeddings for ${uncachedItems.length} items`,
    );

    const BATCH_SIZE = 99;
    for (let i = 0; i < uncachedItems.length; i += BATCH_SIZE) {
      const batch = uncachedItems.slice(i, i + BATCH_SIZE);
      console.log(
        `Processing batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(uncachedItems.length / BATCH_SIZE)}`,
      );
      // CHANGED: Use toText function const { embeddings } = await embedMany({   model: google.textEmbeddingModel('gemini-embedding-001'),   values: batch.map((item) => toText(item)), });
      for (let j = 0; j < batch.length; j++) {
        const item = batch[j];
        const embedding = embeddings[j];
        await writeEmbeddingToCache(toText(item), embedding);
        results.push({ item, embedding });
      }
    }
  }

  return results;
}
```

</Spoiler>

#### Making `searchWithEmbeddings` generic

- [ ] Update `searchWithEmbeddings` to accept a generic type `T` and pass the `toText` function through to `loadOrGenerateEmbeddings`.

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: Made function generic with type parameter T
export async function searchWithEmbeddings<T>(
  query: string,
  items: T[], // CHANGED: Generic items instead of emailChunks
  toText: (item: T) => string, // ADDED: Conversion function
) {
  // CHANGED: Pass toText function to loadOrGenerateEmbeddings
  const embeddings = await loadOrGenerateEmbeddings(
    items,
    toText,
  );

  const { embedding: queryEmbedding } = await embed({
    model: google.textEmbeddingModel('gemini-embedding-001'),
    value: query,
  });

  // CHANGED: Return generic item instead of email
  const results = embeddings.map(({ item, embedding }) => {
    const score = cosineSimilarity(queryEmbedding, embedding);
    return { score, item };
  });

  return results.sort((a, b) => b.score - a.score);
}
```

</Spoiler>

#### Making `reciprocalRankFusion` generic

- [ ] Update `reciprocalRankFusion` to accept a generic type `T` and a `toId` function to generate unique identifiers for items.

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: Made function generic with type parameter T
export function reciprocalRankFusion<T>(
  rankings: { item: T; score: number }[][], // CHANGED: Generic item
  toId: (item: T) => string, // ADDED: Function to get unique ID
): { item: T; score: number }[] {
  const rrfScores = new Map<string, number>();
  const itemMap = new Map<string, T>();

  rankings.forEach((ranking) => {
    ranking.forEach((result, rank) => {
      // CHANGED: Use toId function instead of hardcoded ID logic const itemId = toId(result.item);
      const currentScore = rrfScores.get(itemId) || 0;
      const contribution = 1 / (RRF_K + rank);
      rrfScores.set(itemId, currentScore + contribution);
      itemMap.set(itemId, result.item);
    });
  });

  return Array.from(rrfScores.entries())
    .sort(([, scoreA], [, scoreB]) => scoreB - scoreA)
    .map(([itemId, score]) => ({
      score, // CHANGED: Return generic item item: itemMap.get(itemId)!,
    }));
}
```

</Spoiler>

#### Adding helper functions for email chunks

- [ ] Add `emailChunkToId` helper function to generate unique identifiers for email chunks.

<Spoiler>

```typescript
// src/app/search.ts
// ADDED: Helper to get unique ID for email chunk
export const emailChunkToId = (email: EmailChunk) =>
  `${email.id}-${email.index}`;
```

</Spoiler>

#### Renaming and updating `searchWithRRF`

- [ ] Rename `searchWithRRF` to `searchEmailsWithRRF` and update it to pass the `toText` and `toId` helper functions to the generic search functions.

<Spoiler>

```typescript
// src/app/search.ts
// CHANGED: Renamed function to clarify it searches emails
export const searchEmailsWithRRF = async (
  query: string,
  emails: Email[],
) => {
  const emailChunks = await chunkEmails(emails);
  // CHANGED: Pass emailChunkToText as toText parameter
  const bm25Ranking = await searchWithBM25(
    query.toLowerCase().split(' '),
    emailChunks,
    emailChunkToText,
  );
  // CHANGED: Pass emailChunkToText as toText parameter
  const embeddingsRanking = await searchWithEmbeddings(
    query,
    emailChunks,
    emailChunkToText,
  );
  // CHANGED: Pass emailChunkToId as toId parameter
  const rrfRanking = reciprocalRankFusion(
    [bm25Ranking, embeddingsRanking],
    emailChunkToId,
  );
  return rrfRanking;
};
```

</Spoiler>

#### Updating search tool to use generic functions

- [ ] Import the new helper functions `emailChunkToId` and `emailChunkToText` in the search tool.

```typescript
// src/app/api/chat/search-tool.ts
import {
  chunkEmails,
  emailChunkToId, // ADDED
  emailChunkToText, // ADDED
  loadEmails,
  reciprocalRankFusion,
  searchWithBM25,
  searchWithEmbeddings,
} from '@/app/search';
```

- [ ] Update the search tool to pass `emailChunkToText` to the generic search functions.

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
// CHANGED: Pass emailChunkToText as toText parameter
const bm25Results = keywords
  ? await searchWithBM25(keywords, emailChunks, emailChunkToText)
  : [];
// CHANGED: Pass emailChunkToText as toText parameter
const embeddingResults = searchQuery
  ? await searchWithEmbeddings(
      searchQuery,
      emailChunks,
      emailChunkToText,
    )
  : [];
```

</Spoiler>

- [ ] Update the reciprocal rank fusion call to pass `emailChunkToId`.

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
// CHANGED: Pass emailChunkToId as toId parameter
const rrfResults = reciprocalRankFusion(
  [
    bm25Results.slice(0, NUMBER_PASSED_TO_RERANKER),
    embeddingResults.slice(0, NUMBER_PASSED_TO_RERANKER),
  ],
  emailChunkToId,
);
```

</Spoiler>

- [ ] Update the reranking results mapping to work with the generic `item` property.

<Spoiler>

```typescript
// src/app/api/chat/search-tool.ts
// CHANGED: Map from generic item structure
const rerankedResults = await rerankEmails(
  rrfResults.slice(0, NUMBER_PASSED_TO_RERANKER).map((r) => ({
    email: r.item,
    score: r.score,
  })),
  query,
  conversationHistory,
);
```

</Spoiler>

#### Updating search page

- [ ] Update the search page to import the renamed `searchEmailsWithRRF` function.

```typescript
// src/app/search/page.tsx
// CHANGED: Import renamed function
import { loadEmails, searchEmailsWithRRF } from '../search';
```

- [ ] Update the search page to use the renamed function and handle the generic `item` property.

<Spoiler>

```typescript
// src/app/search/page.tsx
// CHANGED: Use renamed function
const emailsWithScores = await searchEmailsWithRRF(
  query,
  allEmails,
);

// CHANGED: Map from generic item property
const transformedEmails = emailsWithScores
  .map(({ item: email, score }) => ({
    id: email.id,
    from: email.from,
    subject: email.subject,
    preview: email.chunk.substring(0, 100) + '...',
    content: email.chunk,
    chunkIndex: email.index,
    totalChunks: email.totalChunks,
    date: email.timestamp,
    score: score,
  }))
  .sort((a, b) => b.score - a.score);
```

</Spoiler>

## Adding `searchMemories` Function

<!-- VIDEO -->

Let's implement semantic search for memories by creating utility functions to convert messages into searchable queries and a `searchMemories` function that ranks memories by relevance.

Recommendation: hand-code this commit. We saw this as an explainer in the skill-building, so try to build it by hand to make sure you've understood the concepts.

### Steps To Complete

#### Creating `messagePartsToText` and `messageToText` utilities

- [ ] Create a new file `src/app/utils.ts` with functions to convert individual message parts and messages into text format.

These utilities filter out tool calls and focus on the actual text content.

<Spoiler>

```typescript
// src/app/utils.ts
import { MyMessage } from './api/chat/route';

// ADDED: Convert message parts to text, filtering out tool calls/results
export const messagePartsToText = (
  parts: MyMessage['parts'],
) => {
  return parts
    .map((part) => {
      if (part.type === 'text') {
        return part.text;
      }
    })
    .filter((s) => typeof s === 'string')
    .join('\n');
};

// ADDED: Convert a single message to "role: content" format
export const messageToText = (message: MyMessage) => {
  return `${message.role}: ${messagePartsToText(message.parts)}`;
};
```

</Spoiler>

#### Creating the messageHistoryToQuery utility

- [ ] Add the `messageHistoryToQuery` function to `src/app/utils.ts` that converts the entire message history into a semantic query.

This function repeats the most recent message twice to give it extra weight in semantic search rankings.

<Spoiler>

```typescript
// src/app/utils.ts

/**
 * Takes the message history and returns a query that can be
 * used as a semantic search query.
 *
 * Includes the most recent message _twice_ to overweight
 * it in the search results.
 */
// ADDED: Create semantic query by repeating the most recent message
export const messageHistoryToQuery = (messages: MyMessage[]) => {
  const mostRecentMessage = messages[messages.length - 1];

  const query = [...messages, mostRecentMessage]
    .map(messageToText)
    .join('\n');

  return query;
};
```

</Spoiler>

#### Creating the searchMemories function

- [ ] Create a new file `src/app/memory-search.ts` with the `searchMemories` and `memoryToText` functions.

This function loads memories, converts the message history to a semantic query, and ranks memories by relevance using embeddings.

<Spoiler>

```typescript
// src/app/memory-search.ts
// ADDED: Import persistence layer and search utilities
import { DB, loadMemories } from '@/lib/persistence-layer';
import { MyMessage } from './api/chat/route';
import { searchWithEmbeddings } from './search';
import { messageHistoryToQuery } from './utils';

// ADDED: Convert memory to searchable text format
export const memoryToText = (memory: DB.Memory) =>
  `${memory.title}: ${memory.content}`;

// ADDED: Search memories semantically based on message history
export const searchMemories = async (opts: {
  messages: MyMessage[];
}) => {
  // Load all memories from persistence layer
  const memories = await loadMemories();

  // Convert message history to semantic query (with most recent message repeated)
  const query = messageHistoryToQuery(opts.messages);

  // Search and rank memories by relevance to query
  const embeddingsRanking = await searchWithEmbeddings(
    query,
    memories,
    memoryToText,
  );

  return embeddingsRanking;
};
```

</Spoiler>

## Adding Memories to the Agent

<!-- VIDEO -->

We're going to integrate memories into our chat agent so it can recall relevant facts about the user during conversations.

Recommendation: hand-code this commit. Figuring out how to frame the memories in the system prompt is a nice challenge.

### Steps To Complete

#### Import Memory Search Functions

- [ ] Import the memory search functions at the top of `src/app/api/chat/route.ts`

```typescript
import {
  memoryToText,
  searchMemories,
} from '@/app/memory-search';
```

#### Add Memories Configuration

- [ ] Add a constant to define how many memories to use in each conversation

<Spoiler>

```typescript
// ADDED: Define how many memories to include per conversation
const MEMORIES_TO_USE = 3;
```

</Spoiler>

#### Search and Slice Memories

- [ ] In the `POST` function, after validating messages, search for relevant memories using the user's messages

<Spoiler>

```typescript
// ADDED: Search for memories related to the conversation
const allMemories = await searchMemories({ messages });

// ADDED: Take only the top memories based on semantic similarity
const memories = allMemories.slice(0, MEMORIES_TO_USE);
```

</Spoiler>

#### Add Memories to System Prompt

- [ ] Update the system prompt to include the retrieved memories. Insert this section after the `<rules>` section. Use the `memoryToText` function to convert the memories to text.

<Spoiler>

```txt
// ADDED: Include relevant memories in the system prompt
<memories>
Here are some memories that may be relevant to the conversation:

${memories.map((memory) => `- ${memoryToText(memory.item)}`).join("\n")}
</memories>
```

</Spoiler>

#### Testing

- [ ] Add a few test memories to your memory database
- [ ] `console.log` the memory scores to understand which memories are being prioritized
- [ ] Ask the agent questions about user facts (e.g., "What is my name?") to verify it's using the memories correctly

## Added Memory Extraction

<!-- VIDEO -->

Let's implement memory extraction to automatically capture and update permanent information from conversations.

Recommendation: cherry-pick this, then tweak the system prompt to your liking. We've covered most of this in the skill-building, but you can never have too much practice writing prompts.

### Steps To Complete

#### Creating the Memory Extraction Function

- [ ] Create a new file `src/app/api/chat/extract-memories.ts` with imports and the function signature.

```typescript
// src/app/api/chat/extract-memories.ts
import {
  createMemory,
  DB,
  deleteMemory,
  loadMemories,
  updateMemory,
} from '@/lib/persistence-layer';
import { google } from '@ai-sdk/google';
import { convertToModelMessages, generateObject } from 'ai';
import { z } from 'zod';
import { MyMessage } from './route';
import { memoryToText } from '@/app/memory-search';

export async function extractAndUpdateMemories(opts: {
  messages: MyMessage[];
  memories: DB.Memory[];
}) {
  // ADDED: Filter to only user and assistant messages to save costs
  const filteredMessages = opts.messages.filter(
    (message) =>
      message.role === 'user' || message.role === 'assistant',
  );
}
```

- [ ] Add the `generateObject` call with the schema and system prompt to extract memory operations.

<Spoiler>

```typescript
// src/app/api/chat/extract-memories.ts

const memoriesResult = await generateObject({
  model: google('gemini-2.5-flash'),
  schema: z.object({
    updates: z
      .array(
        z.object({
          id: z
            .string()
            .describe('The ID of the existing memory to update'),
          title: z.string().describe('The updated memory title'),
          content: z
            .string()
            .describe('The updated memory content'),
        }),
      )
      .describe('Memories to update'),
    deletions: z
      .array(z.string())
      .describe('Array of memory IDs to delete'),
    additions: z
      .array(
        z.object({
          title: z.string().describe('The memory title'),
          content: z.string().describe('The memory content'),
        }),
      )
      .describe('New memories to add'),
  }),
  system: `You are a memory management agent that extracts and maintains permanent information about the user from conversations.

<existing-memories>
${opts.memories
  .map(
    (memory) =>
      `<memory id="${memory.id}">${memoryToText(memory)}</memory>`,
  )
  .join('\n\n')}
</existing-memories>

Your job is to:
1. Analyze the conversation history
2. Extract NEW permanent facts worth remembering
3. Update existing memories if they should be modified
4. Delete memories that are no longer relevant or accurate

Only store PERMANENT information that:
- Is unlikely to change over time (preferences, traits, characteristics)
- Will be relevant for weeks, months, or years
- Helps personalize future interactions
- Represents lasting facts about the user

Examples of what TO store:
- "User prefers dark mode in applications"
- "User works as a software engineer at Acme Corp"
- "User's primary programming language is TypeScript"
- "User has a cat named Whiskers"

Examples of what NOT to store:
- "User asked about the weather today"
- "User said hello"
- "User is working on a project" (too temporary)
- "User mentioned they're hungry" (temporary state)

For each operation:
- UPDATES: Provide the existing memory ID, new title, and new content
- DELETIONS: Provide memory IDs that are no longer relevant
- ADDITIONS: Provide title and content for brand new memories

Be conservative - only add memories that will genuinely help personalize future conversations.`,
  messages: convertToModelMessages(filteredMessages),
});
```

</Spoiler>

- [ ] Implement the memory operations by extracting the result, filtering conflicts, and executing updates, deletions, and additions.

<Spoiler>

```typescript
// src/app/api/chat/extract-memories.ts

const { updates, deletions, additions } = memoriesResult.object;

// ADDED: Prevent conflicts between updates and deletions
const filteredDeletions = deletions.filter(
  (deletion) =>
    !updates.some((update) => update.id === deletion),
);

// ADDED: Process all memory updates
await Promise.all(
  updates.map((update) =>
    updateMemory(update.id, {
      title: update.title,
      content: update.content,
    }),
  ),
);

// ADDED: Process all memory deletions
await Promise.all(
  filteredDeletions.map((deletion) => deleteMemory(deletion)),
);

// ADDED: Process all memory additions
await Promise.all(
  additions.map((addition) =>
    createMemory({
      id: crypto.randomUUID(),
      title: addition.title,
      content: addition.content,
    }),
  ),
);
```

</Spoiler>

#### Integrating Memory Extraction into Chat Route

- [ ] Import the `extractAndUpdateMemories` function in `src/app/api/chat/route.ts`.

```typescript
import { extractAndUpdateMemories } from './extract-memories';
```

- [ ] Update the memories display format to include memory IDs in the system prompt. This allows the extraction function to reference specific memories to update.

<Spoiler>

```typescript
// src/app/api/chat/route.ts

// CHANGED: Wrap memories in XML tags with IDs for extraction reference
${memories
  .map((memory) => [
    `<memory id="${memory.item.id}">`,
    memoryToText(memory.item),
    "</memory>",
  ])
  .join("\n")}
```

</Spoiler>

- [ ] Add the memory extraction call to the `onFinish` callback. This automatically extracts and updates memories after each conversation turn.

<Spoiler>

```typescript
// src/app/api/chat/route.ts

onFinish: async ({ responseMessage }) => {
  await appendToChatMessages(chatId, [responseMessage]);
  // ADDED: Extract and update memories from the conversation
  await extractAndUpdateMemories({
    messages: [...messages, responseMessage],
    memories: memories.map((memory) => memory.item),
  });
},
```

</Spoiler>

#### Adding Sidebar Refresh on Chat Finish

- [ ] In `src/app/chat.tsx`, add an `onFinish` callback to the `useChat` hook that refreshes the sidebar. This ensures newly extracted memories appear in the UI.

<Spoiler>

```typescript
// src/app/chat.tsx

const { messages, sendMessage, status, regenerate } =
  useChat<MyMessage>({
    id: chatIdInUse,
    messages: props.chat?.messages || [],
    onData: (message) => {
      if (
        message.type === 'data-frontend-action' &&
        message.data === 'refresh-sidebar'
      ) {
        router.refresh();
      }
    },
    // ADDED: Refresh sidebar when chat finishes to show new memories
    onFinish: () => {
      router.refresh();
    },
    generateId: () => crypto.randomUUID(),
  });
```

</Spoiler>

#### Testing the Memory Extraction

- [ ] Run the development server and test by asking questions about yourself.

```bash
pnpm dev
```

- [ ] Try saying "My name is Sarah Chen" or "I live in Chorlton in the UK" and check that memories appear in the sidebar automatically.

## Improved System Prompt

<!-- VIDEO -->

Let's update the system prompt to position the agent as a personal assistant rather than just an email helper. This makes the agent more helpful for general tasks while still supporting email access.

Recommendation: read through the system prompt updates below and see what you want to carry over to your own system prompt.

### Steps To Complete

#### Adding user constants

- [ ] Open `src/app/api/chat/route.ts` and add constants for the user's name at the top of the file, right after the `maxDuration` export.

<Spoiler>

```typescript
// src/app/api/chat/route.ts

// ADDED: User information for personalization
const USER_FIRST_NAME = 'Sarah';
const USER_LAST_NAME = 'Chen';
const MEMORIES_TO_USE = 3;
```

</Spoiler>

#### Updating the task context

- [ ] Update the `<task-context>` section in the system prompt to reflect the agent's new role as a personal assistant.

<Spoiler>

```typescript
// CHANGED: Reframe from email assistant to personal assistant
<task-context>
You are a personal assistant to ${USER_FIRST_NAME} ${USER_LAST_NAME}. You help with general tasks, questions, and can access ${USER_FIRST_NAME}'s email when needed.
</task-context>
```

</Spoiler>

#### Refining the rules

- [ ] Update the rules section to emphasize that the agent should respond naturally to general questions without always using email tools.

<Spoiler>

```typescript
// CHANGED: Add guidance for general vs. email-specific queries
<rules>
- You have THREE email tools available: 'search', 'filterEmails', and 'getEmails'
- Use these tools ONLY when the user explicitly asks about emails or information likely contained in emails
- For general questions, conversations, or tasks unrelated to email, respond naturally without using tools
- When you do need to access emails, follow this multi-step workflow for token efficiency:
```

</Spoiler>

#### Updating the rules footer

- [ ] Add a qualifier to the "NEVER answer from training data" rule so it only applies to email-related queries.

<Spoiler>

```typescript
// CHANGED: Scope the training data rule to email queries only
- For email-related queries, NEVER answer from your training data - always use tools first
```

</Spoiler>

#### Updating the ask section

- [ ] Update the `<the-ask>` section header to clarify that the agent should respond naturally for general questions.

<Spoiler>

```typescript
// CHANGED: Acknowledge both general and email-specific requests
<the-ask>
Here is the user's request. For general questions and conversations, respond naturally. For email-related queries, use the tools and multi-step workflow above.
</the-ask>
```

</Spoiler>

#### Testing the changes

- [ ] Run your chat interface and test with a general greeting like "hello" to verify the agent responds naturally.

- [ ] Test with a personal detail like "I live in Wrexham" to confirm the agent acknowledges it helpfully instead of rejecting it.
