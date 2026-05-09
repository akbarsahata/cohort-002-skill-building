import { google } from '@ai-sdk/google';
import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  streamObject,
  streamText,
  type UIMessage,
} from 'ai';
import { z } from 'zod';
import { searchEmails } from './search.ts';

type MyUIMessage = UIMessage<
  never,
  {
    keywords: {
      keywords: string[];
      searchQuery: string;
    };
  }
>;

export const POST = async (req: Request): Promise<Response> => {
  const body: { messages: MyUIMessage[] } = await req.json();
  const { messages } = body;

  const stream = createUIMessageStream<MyUIMessage>({
    execute: async ({ writer }) => {
      // TODO: Change the generateObject call so that it generates a search query in
      // addition to the keywords. This will be used for semantic search, which will be a
      // big improvement over passing the entire conversation history.
      const keywordPartId = crypto.randomUUID();
      let previousKeywordsPayload = '';

      const keywordsStream = streamObject({
        model: google('gemini-2.5-flash'),
        system: `You are a helpful email assistant, able to search emails for information.
          Your jobs are to generate a list of keywords which will be used to search the emails
          and to generate a search query that will be used for semantic search.
          The keywords should be specific and relevant to the user's question, and should include
          any important terminology or phrases that are likely to be found in the emails.
          The search query should be a natural language query that can be used for semantic search,
          and should capture the intent of the user's question.
        `,
        schema: z.object({
          keywords: z
            .array(z.string())
            .describe(
              'A list of keywords to search the emails with. Use these for exact terminology.',
            ),
          searchQuery: z.string().describe(
            'A natural language search query that captures the intent of the user\'s question. Use this for semantic search.',
          ),
        }),
        messages: convertToModelMessages(messages),
      });

      for await (const partial of keywordsStream.partialObjectStream) {
        const keywordData = {
          keywords:
            partial.keywords?.filter(
              (keyword): keyword is string =>
                typeof keyword === 'string',
            ) ?? [],
          searchQuery: partial.searchQuery ?? '',
        };

        const payloadSignature = JSON.stringify(keywordData);

        if (payloadSignature === previousKeywordsPayload) {
          continue;
        }

        previousKeywordsPayload = payloadSignature;

        writer.write({
          id: keywordPartId,
          type: 'data-keywords',
          data: keywordData,
        });
      }

      const keywords = await keywordsStream.object;

      console.dir(keywords, { depth: null });

      const searchResults = await searchEmails({
        keywordsForBM25: keywords.keywords,
        embeddingsQuery: keywords.searchQuery,
      });

      const topSearchResults = searchResults.slice(0, 5);

      console.log(
        topSearchResults.map((result) => result.email.id),
      );

      const emailSnippets = [
        '## Email Snippets',
        ...topSearchResults.map((result, i) => {
          const from = result.email?.from || 'unknown';
          const to = result.email?.to || 'unknown';
          const subject =
            result.email?.subject || `email-${i + 1}`;
          const body = result.email?.body || '';

          return [
            `### 📧 Email ${i + 1}: [${subject}](#${subject.replace(/[^a-zA-Z0-9]/g, '-')})`,
            `**From:** ${from}`,
            `**To:** ${to}`,
            body,
            '---',
          ].join('\n\n');
        }),
        '## Instructions',
        "Based on the emails above, please answer the user's question. Always cite your sources using the email subject in markdown format.",
      ].join('\n\n');

      const answer = streamText({
        model: google('gemini-2.5-flash'),
        system: `You are a helpful email assistant that answers questions based on email content.
          You should use the provided emails to answer questions accurately.
          ALWAYS cite sources using markdown formatting with the email subject as the source.
          Be concise but thorough in your explanations.
        `,
        messages: [
          ...convertToModelMessages(messages),
          {
            role: 'user',
            content: emailSnippets,
          },
        ],
      });

      writer.merge(answer.toUIMessageStream());
    },
  });

  return createUIMessageStreamResponse({
    stream,
  });
};
