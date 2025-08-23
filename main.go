package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/prompts"
)

func main() {

	// Creamos una implemetación LLM de Ollama
	llm, err := ollama.New(ollama.WithModel("llama3.1:8b"))

	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	history := memory.NewChatMessageHistory()
	systemMessage := "Eres un profesor de secundaria y reponderás de forma fácil a las preguntas que te haga el usuario."

	promptTemplate := prompts.NewChatPromptTemplate([]prompts.MessageFormatter{
		prompts.NewSystemMessagePromptTemplate(
			systemMessage,
			nil,
		),
		prompts.NewGenericMessagePromptTemplate(
			"history",
			"{{range .historyMessages}}{{.GetContent}}\n{{end}}",
			[]string{"historyMessages"},
		),
		prompts.NewHumanMessagePromptTemplate(
			`[Brief] Responde mi pregunta. Esta es mi pregunta: {{.question}}`,
			[]string{"question"},
		),
	})

	scanner := bufio.NewScanner(os.Stdin)

	fmt.Println("Escribe tu consulta (o 'salir' para terminar):")

	for {

		fmt.Print(">>> Consulta: ")

		if !scanner.Scan() {
			break
		}

		question := strings.TrimSpace(scanner.Text())

		if strings.ToLower(question) == "salir" {
			fmt.Println("Saliendo.... Muchas gracias!. Espero haberle ayudado. Saludos.")
			break
		}

		historyMessages, err := history.Messages(ctx)

		if err != nil {
			log.Printf("Error obteniendo historial: %v\n", err)
			continue
		}

		promptText, err := promptTemplate.Format(map[string]any{
			"historyMessages": historyMessages,
			"question":        question,
		})

		if err != nil {
			log.Printf("Error generando prompt: %v\n", err)
			continue
		}

		fmt.Println("Generando respuesta...")

		var respuesta string
		_, err = llms.GenerateFromSinglePrompt(ctx, llm, promptText,
			llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
				fmt.Print(string(chunk))
				respuesta += string(chunk)
				return nil
			}),
		)

		if err != nil {
			log.Printf("Error generando la respuesta: %v\n", err)
			continue
		}

		// Agregar al historial para mantener contexto
		history.AddUserMessage(ctx, question)
		history.AddAIMessage(ctx, respuesta)

		fmt.Println()
	}

}
