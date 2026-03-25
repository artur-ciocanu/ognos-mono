import type { ModelHandle } from "./model-handle.js";

export interface ModelCatalog {
	list(): Promise<ModelHandle[]>;
	resolve(id: string): Promise<ModelHandle | undefined>;
	getDefault(): Promise<ModelHandle | undefined>;
}

export interface StaticModelCatalogOptions {
	defaultModelId?: string;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
	if (typeof value !== "object" || value === null) {
		return false;
	}

	const prototype = Object.getPrototypeOf(value);
	return prototype === Object.prototype || prototype === null;
}

function cloneUnknown<T>(value: T): T {
	try {
		return structuredClone(value);
	} catch {
		if (Array.isArray(value)) {
			return value.map((item) => cloneUnknown(item)) as T;
		}

		if (isPlainObject(value)) {
			const clone: Record<string, unknown> = {};
			for (const [key, entry] of Object.entries(value)) {
				clone[key] = cloneUnknown(entry);
			}
			return clone as T;
		}

		return value;
	}
}

function cloneModelHandle(model: ModelHandle): ModelHandle {
	return {
		...model,
		capabilities: cloneUnknown(model.capabilities),
		raw: cloneUnknown(model.raw),
	};
}

export class StaticModelCatalog implements ModelCatalog {
	readonly #models: ModelHandle[];
	readonly #defaultModelId?: string;

	constructor(models: readonly ModelHandle[], options: StaticModelCatalogOptions = {}) {
		this.#models = models.map((model) => cloneModelHandle(model));
		this.#defaultModelId = options.defaultModelId;
	}

	async list(): Promise<ModelHandle[]> {
		return this.#models.map((model) => cloneModelHandle(model));
	}

	async resolve(id: string): Promise<ModelHandle | undefined> {
		const model = this.#models.find((item) => item.id === id);
		return model ? cloneModelHandle(model) : undefined;
	}

	async getDefault(): Promise<ModelHandle | undefined> {
		if (this.#models.length === 0) {
			return undefined;
		}

		if (this.#defaultModelId) {
			const model = await this.resolve(this.#defaultModelId);
			if (model) {
				return model;
			}
		}

		return cloneModelHandle(this.#models[0]);
	}
}
